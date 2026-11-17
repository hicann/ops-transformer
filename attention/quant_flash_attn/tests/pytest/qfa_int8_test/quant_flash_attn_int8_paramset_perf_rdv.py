# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Model head groups; long cases compare sampled reference rows."""

from .quant_flash_attn_int8_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {}
for heads in (16, 8, 4, 2):
    TEST_PARAMS[f"ViT{heads}"] = {
        "N_q": [heads],
        "N_kv": [heads],
        "D": [72],
        "layout": ["BNSD", "BSND"],
    }
for index, (heads, kv_heads) in enumerate(
    ((24, 4), (12, 2), (6, 1), (16, 2), (8, 1), (4, 1), (32, 2), (16, 1), (64, 4))
):
    TEST_PARAMS[f"GQA{heads}x{kv_heads}"] = {
        "N_q": [heads],
        "N_kv": [kv_heads],
        "D": [64, 128, 256],
        "layout": ["TND"],
        "actual_seq_q": [((6144, 10240, 20480, 40960)[index % 4],)],
    }
for parameters in TEST_PARAMS.values():
    parameters.setdefault("actual_seq_q", [(6144,)])
    parameters.update(
        actual_seq_kv=parameters["actual_seq_q"],
        mask_mode=[0, 3],
        sample_rows=[(0, 127, 511, -1)],
        warmup=[5],
        repeat=[10],
    )

CASES = expand_paramset_to_cases(TEST_PARAMS)
SKIP_CASES = set()
