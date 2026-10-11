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

import sys
import numpy as np


def adapterCapacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_row_idx[i] = -1


def moe_init_routing_v2(
    input_x: np.ndarray,
    expert_idx: np.ndarray,
    active_num: int,
    capacity: int,
    E: int,
    drop_pad_mode: int,
    capacity_flag: bool,
    cumsum_flag: int,
):
    N = input_x.shape[0]
    H = input_x.shape[-1]
    k = expert_idx.shape[-1]
    sorted_row_idx = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable")
    sorted_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)

    expert_tokens_count_or_cumsum = None
    expert_tokens_before_capacity = None

    expert_idx_hist, bins = np.histogram(sorted_expert_idx, bins=E, range=(0, E - 1))
    expert_token_idx = np.cumsum(expert_idx_hist)
    if capacity_flag:
        expert_tokens_before_capacity = expert_idx_hist
    if cumsum_flag == 1:
        expert_tokens_count_or_cumsum = expert_token_idx
    if cumsum_flag == 2:
        expert_tokens_count_or_cumsum = expert_idx_hist

    if drop_pad_mode == 0:
        expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
        expanded_row_idx[sorted_row_idx] = np.arange(
            sorted_row_idx.shape[-1], dtype=np.int32
        )

        active_num = min(active_num, N) * k
        expanded_x = input_x[sorted_row_idx[:active_num] // k, :]
    else:
        adapterCapacity(sorted_row_idx, sorted_expert_idx, capacity)
        sort_row_tmp = np.full((E * capacity), -1, dtype=int)
        offset = 0
        lastExpertId = 0
        for i, val in enumerate(sorted_row_idx):
            if val != -1:
                if lastExpertId != sorted_expert_idx[i]:
                    offset = 0
                    lastExpertId = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * capacity + offset] = sorted_row_idx[
                    i
                ]
                offset = offset + 1

        expanded_row_idx = np.full(sorted_row_idx.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        expanded_x = np.full((E * capacity, H), 0, dtype=input_x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = input_x[val // k]
        expanded_x = expanded_x.reshape(E, capacity, H)

    return (
        expanded_x,
        expanded_row_idx.astype("int32"),
        expert_tokens_count_or_cumsum,
        expert_tokens_before_capacity,
    )


def gen_golden_data_simple(
    N, k, H, active_num, C, E, drop_mode, cumsum_flag, capacity_flag, quant_mode, dtype
):
    N = int(N)
    k = int(k)
    H = int(H)
    active_num = int(active_num)
    C = int(C)
    E = int(E)
    drop_pad_mode = int(drop_mode)
    capacity_flag = bool(capacity_flag)
    cumsum_flag = int(cumsum_flag)
    quant_mode = int(quant_mode)

    input_x = np.random.uniform(-1, 100, [N, H]).astype(dtype)
    input_expertIdx = np.random.randint(0, E, size=(N, k)).astype("int32")
    if quant_mode == 0:
        scale = np.random.randn(1).astype(np.float32)
        offset = np.random.randn(1).astype(np.float32)
    else:
        scale = np.random.uniform(-1, 100, [1, H]).astype(np.float32)
    # expandedX, expandedRowIdx, expert_tokens_count_or_cumsum, expert_tokens_before_capacity = moe_init_routing_v2(input_x,
    #     input_expertIdx, active_num, C, E, drop_pad_mode, capacity_flag, cumsum_flag)

    input_x.tofile("./input_x.bin")
    input_expertIdx.tofile("./input_expertIdx.bin")
    scale.tofile("./scale.bin")
    if quant_mode == 0:
        offset.tofile("./offset.bin")

    # expandedX.tofile("./golden_0.bin")
    # expandedRowIdx.tofile("./golden_1.bin")
    # expandedExpertIdx.tofile("./golden_2.bin")


# python3 gen_data.py 8 2 5120 8 float32
if __name__ == "__main__":
    gen_golden_data_simple(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
        sys.argv[8],
        sys.argv[9],
        sys.argv[10],
        sys.argv[11],
    )
