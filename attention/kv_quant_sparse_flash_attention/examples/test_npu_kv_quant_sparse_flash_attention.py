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
import random
import math

# 生成随机数据，并发送到npu
query_type = torch.bfloat16
scale_value = 0.041666666666666664
sparse_block_size = 1
sparse_block_count = 2048
b = 1
s1 = 1
s2 = 8192
n1 = 128
n2 = 1
dn = 512
dr = 64
tile_size = 128
block_size = 256
layout_query = "BSND"
s2_act = 4096

query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
key = torch.tensor(
    np.random.uniform(-5, 10, (b * (s2 // block_size), block_size, n2, dn))
).to(torch.int8)
value = key.clone().npu()
idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
sparse_indices = (
    torch.tensor([idxs for _ in range(b * s1 * n2)])
    .reshape(b, s1, n2, sparse_block_count)
    .to(torch.int32)
    .npu()
)
query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
key_rope = torch.tensor(
    np.random.uniform(-10, 10, (b * (s2 // block_size), block_size, n2, dr))
).to(query_type)
act_seq_q = torch.tensor([s1] * b).to(torch.int32).npu()
act_seq_kv = torch.tensor([s2_act] * b).to(torch.int32).npu()
antiquant_scale = torch.tensor(
    np.random.uniform(
        -100, 100, (b * (s2 // block_size), block_size, n2, dn // tile_size)
    )
).to(torch.float32)
key = torch.cat(
    (key, key_rope.view(torch.int8), antiquant_scale.view(torch.int8)), axis=3
).npu()
query = torch.cat((query, query_rope), axis=3).npu()
block_table = (
    torch.tensor([range(b * s2 // block_size)], dtype=torch.int32).reshape(b, -1).npu()
)

# 调用qsfa算子
out = torch_npu.npu_kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value=scale_value,
    sparse_block_size=sparse_block_size,
    actual_seq_lengths_query=act_seq_q,
    actual_seq_lengths_kv=act_seq_kv,
    layout_query="BSND",
    layout_kv="PA_BSND",
    sparse_mode=3,
    block_table=block_table,
    attention_mode=2,
    quant_scale_repo_mode=1,
    tile_size=tile_size,
    key_quant_mode=2,
    value_quant_mode=2,
    rope_head_dim=64,
)
