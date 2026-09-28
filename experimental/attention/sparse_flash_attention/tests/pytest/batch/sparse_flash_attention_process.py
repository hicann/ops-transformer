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
import torch_npu


def call_npu(input_data):
    """将输入数据搬移到NPU并调用torch_npu.sparse_flash_attention，返回算子结果。"""
    tensor_input = input_data["input"]
    attr = input_data["attr"]

    query = tensor_input["query"].npu()
    key = tensor_input["key"].npu()
    value = tensor_input["value"].npu()
    sparse_indices = tensor_input["sparse_indices"].npu()
    block_table = tensor_input["block_table"]
    if block_table is not None:
        block_table = block_table.npu()
    cur_seq_lengths_query = tensor_input["cur_seq_lengths_query"].npu()
    cur_seq_lengths_kv = tensor_input["cur_seq_lengths_kv"].npu()
    query_rope = tensor_input["query_rope"].npu()
    key_rope = tensor_input["key_rope"].npu()

    npu_result, softmax_max, softmax_sum = torch_npu.sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        attr["scale_value"],
        block_table=block_table,
        cur_seq_lengths_query=cur_seq_lengths_query,
        cur_seq_lengths_kv=cur_seq_lengths_kv,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=attr["sparse_block_size"],
        layout_query=attr["layout_query"],
        layout_kv=attr["layout_kv"],
        sparse_mode=attr["sparse_mode"],
        return_softmax_lse=attr["return_softmax_lse"],
        return_float_output=attr["return_float_output"],
    )

    torch.npu.synchronize()
    return npu_result, softmax_max, softmax_sum
