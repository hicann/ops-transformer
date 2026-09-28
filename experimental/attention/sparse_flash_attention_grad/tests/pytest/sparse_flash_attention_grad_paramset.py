#!/usr/bin/python
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

import torch

# 定义测试参数组合
# params结构:
# (layout, dtype, seqlen_q, seqlen_kv, n1, n2, head_dim, rope_head_dim,
#  sparse_block_size, sparse_block_count, sparse_mode, deterministic, case_name)
TEST_PARAMS = {
    "tnd_fp16_basic": {
        "layout": ["TND"],
        "dtype": [torch.float16],
        "seqlen_q": [[4, 4]],
        "seqlen_kv": [[500, 600]],
        "n1": [32],
        "n2": [1],
        "head_dim": [512],
        "rope_head_dim": [64],
        "sparse_block_size": [1],
        "sparse_block_count": [256],
        "sparse_mode": [3],
        "deterministic": [False],
    },
    "tnd_bf16_basic": {
        "layout": ["TND"],
        "dtype": [torch.bfloat16],
        "seqlen_q": [[4, 4]],
        "seqlen_kv": [[500, 600]],
        "n1": [32],
        "n2": [1],
        "head_dim": [512],
        "rope_head_dim": [64],
        "sparse_block_size": [1],
        "sparse_block_count": [256],
        "sparse_mode": [3],
        "deterministic": [False],
    },
    "bsnd_fp16_basic": {
        "layout": ["BSND"],
        "dtype": [torch.float16],
        "seqlen_q": [[2, 2]],
        "seqlen_kv": [[200, 220]],
        "n1": [32],
        "n2": [1],
        "head_dim": [512],
        "rope_head_dim": [64],
        "sparse_block_size": [1],
        "sparse_block_count": [256],
        "sparse_mode": [3],
        "deterministic": [False],
    },
    "tnd_fp16_no_rope": {
        "layout": ["TND"],
        "dtype": [torch.float16],
        "seqlen_q": [[2, 2]],
        "seqlen_kv": [[200, 220]],
        "n1": [32],
        "n2": [1],
        "head_dim": [512],
        "rope_head_dim": [0],
        "sparse_block_size": [1],
        "sparse_block_count": [256],
        "sparse_mode": [3],
        "deterministic": [False],
    },
    "tnd_fp16_full_sparse": {
        "layout": ["TND"],
        "dtype": [torch.float16],
        "seqlen_q": [[2, 2]],
        "seqlen_kv": [[200, 220]],
        "n1": [32],
        "n2": [1],
        "head_dim": [512],
        "rope_head_dim": [64],
        "sparse_block_size": [1],
        "sparse_block_count": [256],
        "sparse_mode": [0],
        "deterministic": [True],
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = []


def build_params():
    for key in TEST_PARAMS.keys():
        params = TEST_PARAMS[key]
        case_name = key
        ENABLED_PARAMS.append(
            (
                params["layout"][0],
                params["dtype"][0],
                params["seqlen_q"][0],
                params["seqlen_kv"][0],
                params["n1"][0],
                params["n2"][0],
                params["head_dim"][0],
                params["rope_head_dim"][0],
                params["sparse_block_size"][0],
                params["sparse_block_count"][0],
                params["sparse_mode"][0],
                params["deterministic"][0],
                case_name,
            )
        )


build_params()
