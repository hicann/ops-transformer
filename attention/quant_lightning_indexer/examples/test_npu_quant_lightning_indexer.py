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
import numpy as np
import torch.nn as nn
import math

n1 = 64
n2 = 1
d = 128
block_size = 128
layout_key = "PA_BSND"
layout_query = "BSND"
query_quant_mode = 0
key_quant_mode = 0
np.random.seed(0)
# -------------
b = 24
t = None
s1 = 4
s2 = 512
act_seq_q = None
act_seq_k = None
sparse_mode = 0
sparse_count = 2048
max_block_table_num = (s2 + block_size - 1) // block_size
block_table = torch.tensor([range(b * max_block_table_num)], dtype=torch.int32).reshape(
    b, -1
)
key = torch.tensor(
    np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))
).to(torch.int8)
key_dequant_scale = torch.tensor(
    np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2))
)
key_dequant_scale = key_dequant_scale.to(torch.float16)
query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(
    torch.float16
)
weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
actual_seq_lengths_query = (
    torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32)
    if act_seq_q is None
    else torch.tensor(act_seq_q).to(torch.int32)
)
actual_seq_lengths_key = (
    torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32)
    if act_seq_k is None
    else torch.tensor(act_seq_k).to(torch.int32)
)

npu_out = torch_npu.npu_quant_lightning_indexer(
    query.npu(),
    key.npu(),
    weights.npu(),
    query_dequant_scale.npu(),
    key_dequant_scale.npu(),
    actual_seq_lengths_query=actual_seq_lengths_query.npu(),
    actual_seq_lengths_key=actual_seq_lengths_key.npu(),
    block_table=block_table.npu(),
    query_quant_mode=query_quant_mode,
    key_quant_mode=key_quant_mode,
    layout_query=layout_query,
    layout_key=layout_key,
    sparse_count=sparse_count,
    sparse_mode=sparse_mode,
)
