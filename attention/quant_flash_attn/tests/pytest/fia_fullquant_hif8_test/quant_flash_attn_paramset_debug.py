# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from quant_flash_attn_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "TND_B1_QS4_KVS1024_Nq64_Nkv8_D128_SP3": {
        "B": [1],
        "N_q": [64],
        "N_kv": [8],
        "D": [128],
        "cu_seqlens_q": [[0, 4]],
        "cu_seqlens_kv": [[0, 1024]],
        "seqused_q": [[4]],
        "seqused_kv": [[1024]],
        "max_seqlen_q": [4],
        "max_seqlen_kv": [1024],
        "mask_mode": [3],
        "q_scale_layout": ["BSND"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    "TND_B1_QS128_KVS1024_Nq64_Nkv8_D128_SP3": {
        "B": [1],
        "N_q": [64],
        "N_kv": [8],
        "D": [128],
        "cu_seqlens_q": [[0, 128]],
        "cu_seqlens_kv": [[0, 1024]],
        "seqused_q": [[128]],
        "seqused_kv": [[1024]],
        "max_seqlen_q": [128],
        "max_seqlen_kv": [1024],
        "mask_mode": [3],
        "q_scale_layout": ["BSND"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
