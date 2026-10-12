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
TEST_PARAMS = {
    "li_default_a2": {  # 1
        "batch_size": [18],
        "q_seq": [3],
        "k_seq": [3072],
        "q_t_size": [1],
        "k_t_size": [1],  # 压缩后的值
        "q_head_num": [16],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128],  # 取16的整数倍，最多支持到1024
        "block_num": [469],
        "qk_dtype": [torch.bfloat16],
        "weight_dtype": [torch.bfloat16],
        "actual_seq_dtype": [torch.int32],
        "act_seq_q": [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]],
        "act_seq_k": [
            [
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3016,
                3072,
            ]
        ],  # PA场景非前缀和，表示每个batch_size的实际token数
        "layout_query": ["BSND"],
        "layout_key": ["PA_BSND"],
        "sparse_count": [2048],
        "sparse_mode": [3],
        "query_datarange": [[-1, 1]],
        "key_datarange": [[-1, 1]],
        "weights_datarange": [[-130, 130]],
    },
    "li_default_a3": {},
    "li_default_a5": {},
}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
# 定义设备映射
device_configs = {
    "Ascend910_93": "li_default_a3",
    "Ascend950": "li_default_a5",
    "Ascend910B": "li_default_a2",
}
# 选择配置 key
config_key = device_configs.get(properties.name)
if config_key:
    ENABLED_PARAMS = [TEST_PARAMS[config_key]]
else:
    print(f"未知设备: {properties.name}, 使用默认配置（li_default_a2）")
    ENABLED_PARAMS = [TEST_PARAMS["li_default_a2"]]
