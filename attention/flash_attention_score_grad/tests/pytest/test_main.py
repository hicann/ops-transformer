#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import torch_npu
import numpy as np
import pytest
from common import CalculationContext, log
from test_case import TestCases
from cpu_impl import attentionScoreWithGrad, run_unpad
from npu_impl import PTAtest


def checkResult(a, b, name):
    log(f"info: 开始计算 {name} 精度。")
    ratio_threshold = 0.005
    threshold_diff = 0.005
    if a.shape == b.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(a, b)):
            log(f"info: {name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            ratio = 1
            max = 0
            sum = 0
        else:
            # 统计两个张量的值差距超过5‰的元素数量和占比
            diff = torch.abs(a.sub(b))
            min = torch.empty(a.shape, dtype=torch.float16).to(a.device)
            min.fill_(0.000025)
            max = torch.max(torch.abs(a), torch.abs(b))
            threshold = torch.max(max.mul(ratio_threshold), min)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(a)
            if torch.max(diff) > 0.1:
                log(
                    f"error: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%}，超过阈值，用例执行失败！！！"
                )
            else:
                log(
                    f"info: {name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%} 。"
                )
            ratio = (1 - ratio_diff).cpu().detach().numpy()
            # log(a.max(), b.max())
            max = torch.max(diff).cpu().detach().numpy()
            sum = torch.sum(diff).cpu().detach().numpy()
        log(f"{name} diff_max : {max:.8f}")
        log(f"{name} diff_sum : {sum:.6f}")
    else:
        log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        log(f"debug: a {a.shape}")
        log(f"debug: b {b.shape}")
        ratio = 0
        max = 999999
        sum = 999999
    return ratio, max, sum


def test_npu_flash_atten_grad():
    for test_name, test_data in TestCases.items():
        print(f"=============Run test case: {test_name} begin ============")
        ctx = CalculationContext(input_case=test_data)
        func = (
            run_unpad if test_data["input_layout"] == "TND" else attentionScoreWithGrad
        )
        golden_tuple = func(ctx)
        npu_tuple = PTAtest(ctx)
        if test_data["rope"] == 0:
            names = ["dq", "dk", "dv"]
        else:
            names = ["dq", "dk", "dv", "dq_rope", "dk_rope"]
        for name, g_val, n_val in zip(names, golden_tuple, npu_tuple):
            device = n_val.device
            checkResult(
                n_val.float().flatten(),
                g_val.float().to(device).flatten(),
                f"{test_name} {name}",
            )
        print(f"=============Run test case: {test_name} end ============")
