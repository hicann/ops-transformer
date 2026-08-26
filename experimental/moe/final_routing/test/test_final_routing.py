#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import torch
import torch_npu
import npu_ops_transformer_ext

logging.basicConfig(level=logging.INFO)

TOKEN_NUM = 2
EXPERT_NUM = 4
HIDDEN_SIZE = 8
BLOCK_DIM = 40

input_tensor = torch.randint(1, 10, (TOKEN_NUM * EXPERT_NUM, HIDDEN_SIZE)).to(torch.bfloat16)
token_table = torch.arange(TOKEN_NUM * EXPERT_NUM, dtype=torch.int32).reshape(TOKEN_NUM, EXPERT_NUM)
score_table = torch.randint(0, 10, (TOKEN_NUM, EXPERT_NUM)).to(torch.bfloat16)
output = torch.empty(TOKEN_NUM, HIDDEN_SIZE, dtype=torch.bfloat16)
output_cpu = torch.empty_like(output).float()

input_32 = input_tensor.float()
for t in range(TOKEN_NUM):
    for e in range(EXPERT_NUM):
        input_idx = token_table[t, e].item()
        if input_idx < 0:
            continue
        score = score_table[t, e].item()
        feat = input_32[input_idx]
        output_cpu[t] += feat * score

output_cpu = output_cpu.to(torch.bfloat16)
output_npu = output.npu()
torch.ops.npu_ops_transformer_ext.final_routing(BLOCK_DIM, input_tensor.npu(), token_table.npu(),
    score_table.npu(), output_npu)

logging.info(f"cpu result vs npu result: {torch.equal(output_cpu, output_npu.cpu())}")