#!/usr/bin/python3
# -*- coding:utf-8 -*-
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
import os
import numpy as np
import torch


def gen_data(
    row,
    hidden,
    topk,
    has_scales,
    has_bias,
    drop_pad_mode,
    active_num,
    expert_num,
    expert_capacity,
    dtype1,
    dtype2,
):
    grad_y_data = np.random.uniform(-100, 100, [int(row), int(hidden)]).astype(dtype1)
    grad_y_data.tofile("./grad_y.bin")

    if drop_pad_mode == 0:
        if active_num > 0 and active_num < row * topk:
            expanded_row_idx_data = np.random.choice(
                active_num, row * topk, replace=True
            ).astype(dtype2)
            expanded_row_idx_data.tofile("./expanded_row_idx.bin")

            if has_scales == True:
                expanded_x_data = np.random.uniform(
                    -100, 100, [int(active_num), int(hidden)]
                ).astype(dtype1)
                expanded_x_data.tofile("./expanded_x.bin")
        else:
            expanded_row_idx_data = np.random.choice(
                row * topk, row * topk, replace=False
            ).astype(dtype2)
            expanded_row_idx_data.tofile("./expanded_row_idx.bin")

            if has_scales == True:
                expanded_x_data = np.random.uniform(
                    -100, 100, [int(row * topk), int(hidden)]
                ).astype(dtype1)
                expanded_x_data.tofile("./expanded_x.bin")
    else:
        expanded_row_idx_old_data = np.random.uniform(
            -1,
            expert_num * expert_capacity,
            [
                int(row * topk),
            ],
        ).astype(dtype2)
        expanded_row_idx_unique_data, index = np.unique(
            expanded_row_idx_old_data, return_index=True
        )
        expanded_row_idx_data = np.full_like(expanded_row_idx_old_data, -1)
        expanded_row_idx_data[index] = expanded_row_idx_unique_data
        expanded_row_idx_data.tofile("./expanded_row_idx.bin")

        if has_scales == True:
            expanded_x_data = np.random.uniform(
                -100, 100, [int(expert_num), int(expert_capacity), int(hidden)]
            ).astype(dtype1)
            expanded_x_data.tofile("./expanded_x.bin")

    if has_scales == True:
        scales_data = np.random.uniform(-100, 100, [int(row), int(topk)]).astype(dtype1)
        scales_data.tofile("./scales.bin")

        if has_bias == True:
            expert_idx_data = np.random.choice(
                expert_num, row * topk, replace=True
            ).astype(dtype2)
            expert_idx_data.tofile("./expert_idx.bin")

            bias_data = np.random.uniform(
                -100, 100, [int(expert_num), int(topk)]
            ).astype(dtype1)
            bias_data.tofile("./bias.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_data(
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
    exit(0)
