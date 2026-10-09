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
    "TND_B1_QS128_KVS2048_Nq1_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "cu_seqlens_q": [[0, 128]],
        "cu_seqlens_kv": [[0, 2048]],
        "seqused_q": [[128]],
        "seqused_kv": [[2048]],
        "max_seqlen_q": [128],
        "max_seqlen_kv": [2048],
        "mask_mode": [3],
        "q_scale_layout": ["BSND"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    "TND_B1_QS512_KVS5120_Nq80_Nkv2_D128_SP3": {
        "B": [1],
        "N_q": [80],
        "N_kv": [2],
        "D": [128],
        "cu_seqlens_q": [[0, 512]],
        "cu_seqlens_kv": [[0, 5120]],
        "seqused_q": [[512]],
        "seqused_kv": [[5120]],
        "max_seqlen_q": [512],
        "max_seqlen_kv": [5120],
        "mask_mode": [3],
        "q_scale_layout": ["BSND"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
    "TND_B1_QS3000_KVS6000_Nq16_Nkv8_D128_SP3": {
        "B": [1],
        "N_q": [16],
        "N_kv": [8],
        "D": [128],
        "cu_seqlens_q": [[0, 3000]],
        "cu_seqlens_kv": [[0, 6000]],
        "seqused_q": [[3000]],
        "seqused_kv": [[6000]],
        "max_seqlen_q": [3000],
        "max_seqlen_kv": [6000],
        "mask_mode": [3],
        "q_scale_layout": ["BSND"],
        "p_scale": [1.0],
        "enable_lse": [False],
    },
}

# Long-sequence GQA performance cases share all parameters except sequence length.
for seq_len in (10240, 16384, 20480, 32768, 65536):
    TEST_PARAMS[f"TND_B1_QS{seq_len}_KVS{seq_len}_Nq20_Nkv2_D128_SP0"] = {
        "B": [1],
        "N_q": [20],
        "N_kv": [2],
        "D": [128],
        "cu_seqlens_q": [[0, seq_len]],
        "cu_seqlens_kv": [[0, seq_len]],
        "seqused_q": [[seq_len]],
        "seqused_kv": [[seq_len]],
        "max_seqlen_q": [seq_len],
        "max_seqlen_kv": [seq_len],
        "mask_mode": [0],
        "q_scale_layout": ["BSND"],
        "p_scale": [1.0],
        "enable_lse": [False],
        "input_layout": ["TND"],
    }

CASES = expand_paramset_to_cases(TEST_PARAMS)
