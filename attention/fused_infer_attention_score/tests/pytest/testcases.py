#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
    # 基础场景
    "base_default": {
        "cache_layout": ["BBH"],
        "block_size": [128],
        "in_layout": ["BNSD"],
        "dtype": [torch.float16],
        "head_dim": [128],  # 以下为必选
        "kv_seq": [16384],
        "q_seq": [32],  # >1,PFA
        "kv_head_num": [4],
        "q_head_num": [8],
        "batch_size": [2],
    },
    "GQA_no_quant": {
        "cache_layout": ["BBH"],
        "block_size": [128, 256],
        "in_layout": ["BNSD", "BSND"],
        "dtype": [torch.float16],
        "head_dim": [128],  # 以下为必选
        "kv_seq": [16384],
        "q_seq": [1, 64, 128],  # >1,PFA
        "kv_head_num": [4, 8, 16],
        "q_head_num": [16, 32],
        "batch_size": [4, 8],
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [
    TEST_PARAMS["base_default"]
]  # 或指定子集，如[TEST_PARAMS["base"], TEST_PARAMS["mixed"]], TEST_PARAMS
GRAPH_MODE_PARAMS = [TEST_PARAMS["base_default"]]
