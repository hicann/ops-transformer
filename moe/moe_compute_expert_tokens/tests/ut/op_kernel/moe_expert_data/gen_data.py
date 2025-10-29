#!/usr/bin/python3
# -*- coding:utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


import sys
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")


def compute_total_rows_before_expert(
        sorted_experts:np.ndarray, 
        num_experts: int) -> np.ndarray:
    arr_length = sorted_experts.shape[-1]
    res = np.arange(num_experts)
    for i in range(num_experts):
        target = i
        low = 0
        high = arr_length - 1
        target_location = -1
        while low <= high:
            mid = (low + high) // 2
            if sorted_experts[mid] > target:
                high = mid - 1
            else:
                low = mid + 1
                target_location = mid
        res[i] = target_location + 1
    return res


def general_input_data(sorted_expert_len, experts_num):
    random_int_list = []
    for _ in range(sorted_expert_len):
        random_int_list.append(random.randint(0, sorted_expert_len))
    sorted_experts = np.sort(random_int_list)
    num_experts = experts_num
    return sorted_experts, num_experts


def gen_golden_data_simple(BSK_len, num_expert):
    sorted_experts, num_experts = general_input_data(BSK_len, num_expert)
    result = compute_total_rows_before_expert(
        sorted_experts, num_experts
    )
    sorted_experts.astype(np.int32).tofile("./sorted_experts.bin")
    np.array(num_experts).astype(np.int32).tofile("./num_experts.bin")
    result.astype(np.int32).tofile("./output_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2])


