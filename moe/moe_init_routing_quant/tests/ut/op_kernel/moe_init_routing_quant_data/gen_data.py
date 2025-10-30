#!/usr/bin/python
# -*- coding: utf-8 -*-
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

def ascend_quant_cpu(x, scale, offset):
    cast_fp16 = x.astype("float16")
    scale_rst = cast_fp16 * scale
    add_offset = scale_rst + offset
    round_data = np.rint(add_offset)
    round_data = np.clip(round_data, -128, 127)
    output = round_data.astype("int8")
    return output

def moe_init_routing_quant(input_activations: np.ndarray,
                    source_rows: np.ndarray,
                    expert_for_source_row: np.ndarray,
                    active_rows: int,
                    scale,
                    offset) -> (np.ndarray, np.ndarray, np.ndarray):
    num_rows = input_activations.shape[0]
    hidden_size = input_activations.shape[-1]
    k = expert_for_source_row.shape[-1]
    sort_expert_for_source_row = np.argsort(expert_for_source_row.reshape((-1,)), axis=-1, kind="stable")
    new_expert_for_source_row = np.sort(expert_for_source_row.reshape((-1,)), axis=-1)

    expanded_dst_to_src_row = np.take_along_axis(source_rows.reshape((-1,)), sort_expert_for_source_row, axis=-1)
    expanded_src_to_dst_row = np.zeros(expanded_dst_to_src_row.shape)
    expanded_src_to_dst_row[expanded_dst_to_src_row] = np.arange(expanded_dst_to_src_row.shape[-1])

    active_rows = min(active_rows, num_rows) * k
    expanded_activations = input_activations[expanded_dst_to_src_row[:active_rows] % num_rows, :]

    expanded_activations = ascend_quant_cpu(expanded_activations, scale, offset)
    return expanded_activations, expanded_src_to_dst_row.astype("int32"), new_expert_for_source_row.astype("int32")


def gen_golden_data_simple(num_rows, k, hidden_size, active_rows, scale, offset, dtype):
    num_rows = int(num_rows)
    k = int(k)
    hidden_size = int(hidden_size)
    active_rows = int(active_rows)
    scale = float(scale)
    offset = float(offset)

    input_x = np.random.uniform(-1, 100, [num_rows, hidden_size]).astype(dtype)
    input_rowIdx = np.arange(num_rows * k).reshape([k, num_rows]).transpose(1, 0).astype("int32")
    input_expertIdx = np.random.randint(0, 100, size=(num_rows, k)).astype("int32")

    expandedX, expandedRowIdx, expandedExpertIdx  = moe_init_routing_quant(input_x, input_rowIdx, input_expertIdx, active_rows, scale, offset)

    input_x.tofile("./input_x.bin")
    input_rowIdx.tofile("./input_rowIdx.bin")
    input_expertIdx.tofile("./input_expertIdx.bin")

    # expandedX.tofile("./golden_0.bin")
    # expandedRowIdx.tofile("./golden_1.bin")
    # expandedExpertIdx.tofile("./golden_2.bin")

# python3 gen_data.py 8 2 5120 8 float32
if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])