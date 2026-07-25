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
    # ======================================================================
    # Prefill: 大 batch + 长序列
    # ======================================================================
    "B1_G8_Nq16_Nkv2_D128_SM3_LSE0_Q8192_KVS8192_Prefill": {
        "B": [1],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[8192]],
        "actual_seq_kv": [[8192]],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B2_G8_Nq16_Nkv2_D128_SM3_LSE0_Q4096_KVS8192_Prefill": {
        "B": [2],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[4096, 4096]],
        "actual_seq_kv": [[8192, 8192]],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B4_G8_Nq16_Nkv2_D128_SM3_LSE0_Q4096_KVS8192_Prefill": {
        "B": [4],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[4096] * 4],
        "actual_seq_kv": [[8192] * 4],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B8_G8_Nq16_Nkv2_D128_SM3_LSE0_Q2048_KVS4096_Prefill": {
        "B": [8],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[2048] * 8],
        "actual_seq_kv": [[4096] * 8],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B16_G8_Nq16_Nkv2_D128_SM3_LSE0_Q2048_KVS2048_Prefill": {
        "B": [16],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[2048] * 16],
        "actual_seq_kv": [[2048] * 16],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B32_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1024_KVS2048_Prefill": {
        "B": [32],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[1024] * 32],
        "actual_seq_kv": [[2048] * 32],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    # ======================================================================
    # Decode: 大 batch + 短 Q + 长 KV
    # ======================================================================
    "B64_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1_KVS8192_Decode": {
        "B": [64],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[1] * 64],
        "actual_seq_kv": [[8192] * 64],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B128_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1_KVS8192_Decode": {
        "B": [128],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[1] * 128],
        "actual_seq_kv": [[8192] * 128],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B128_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1_KVS16384_Decode": {
        "B": [128],
        "N_q": [16],
        "N_kv": [2],
        "actual_seq_q": [[1] * 128],
        "actual_seq_kv": [[16384] * 128],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    # ======================================================================
    # 高 GQA 比例 + 长 KV
    # ======================================================================
    "B1_G64_Nq64_Nkv1_D128_SM3_LSE0_Q1_KVS16384_HighGQA": {
        "B": [1],
        "N_q": [64],
        "N_kv": [1],
        "actual_seq_q": [[1]],
        "actual_seq_kv": [[16384]],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B1_G128_Nq128_Nkv1_D128_SM3_LSE0_Q1_KVS16384_HighGQA": {
        "B": [1],
        "N_q": [128],
        "N_kv": [1],
        "actual_seq_q": [[1]],
        "actual_seq_kv": [[16384]],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    # ======================================================================
    # 极长序列
    # ======================================================================
    "B1_G4_Nq16_Nkv4_D128_SM3_LSE0_Q16384_KVS16384_VLong": {
        "B": [1],
        "N_q": [16],
        "N_kv": [4],
        "actual_seq_q": [[16384]],
        "actual_seq_kv": [[16384]],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
    "B1_G2_Nq16_Nkv8_D128_SM3_LSE0_Q1_KVS65536_VLong": {
        "B": [1],
        "N_q": [16],
        "N_kv": [8],
        "actual_seq_q": [[1]],
        "actual_seq_kv": [[65536]],
        "sparse_mode": [3],
        "enable_lse": [False],
        "p_scale": [1.0],
    },
}

SKIP_CASES = {
    "B128_G8_Nq16_Nkv2_D128_SM3_LSE0_Q1_KVS16384_Decode",
    "B1_G2_Nq16_Nkv8_D128_SM3_LSE0_Q1_KVS65536_VLong",
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
