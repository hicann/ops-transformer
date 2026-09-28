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
    """调用torch_npu.sparse_flash_attention_grad，返回算子输出的梯度结果。"""
    query = input_data["query"]
    key = input_data["key"]
    value = input_data["value"]
    out = input_data["out"]
    dout = input_data["dout"]
    sparse_indices = input_data["sparse_indices"]
    query_rope = input_data["query_rope"]
    key_rope = input_data["key_rope"]
    softmax_max = input_data["softmax_max"]
    softmax_sum = input_data["softmax_sum"]
    cur_seq_lengths_query = input_data["cur_seq_lengths_query"]
    cur_seq_lengths_kv = input_data["cur_seq_lengths_kv"]

    scale_value = input_data["scale_value"]
    sparse_block_size = input_data["sparse_block_size"]
    layout = input_data["layout"]
    sparse_mode = input_data["sparse_mode"]
    deterministic = input_data["deterministic"]

    npu_results = torch_npu.sparse_flash_attention_grad(
        query,
        key,
        value,
        sparse_indices,
        dout,
        out,
        softmax_max,
        softmax_sum,
        query_rope=query_rope,
        key_rope=key_rope,
        cur_seq_lengths_query=cur_seq_lengths_query,
        cur_seq_lengths_kv=cur_seq_lengths_kv,
        scale_value=scale_value,
        sparse_block_size=sparse_block_size,
        layout=layout,
        sparse_mode=sparse_mode,
        deterministic=deterministic,
    )

    torch.npu.synchronize()
    return npu_results
