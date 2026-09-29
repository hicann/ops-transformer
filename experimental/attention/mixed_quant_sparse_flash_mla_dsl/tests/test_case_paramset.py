# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

TEST_PARAMS = {
    "decode": {
        "Testcase_Name": [None],
        "template_run_mode": ["ORI_CMP_SPARSE"],
        "B": [12],
        "S1": [6],
        "S2": [128 * 1024],
        "S2C": [64 * 1024],
        "K1": [128],
        "K2": [512],
        "ori_kv_topk_mode": ["fullK"],
        "cmp_kv_topk_mode": ["fullK"],
        "block_size1": [128],
        "block_size2": [128],
        "seed": [7],
        "kv_axis0_noncontiguous": [False],
    },
}

ENABLED_PARAMS = list(TEST_PARAMS.values())
