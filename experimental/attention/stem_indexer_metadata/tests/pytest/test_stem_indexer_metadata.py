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
import torchair
import torch.nn as nn
import npu_ops_transformer
import pandas as pd

AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
METADATA_HEADER_SIZE = 16
FA_METADATA_SIZE = 16
AIV_RESERVED_METADATA_SIZE = 16
METADATA_ALIGN_SIZE = 4096

qSeqLens = torch.tensor([384], dtype=torch.int32).npu()
kvSeqLens = torch.tensor([32768], dtype=torch.int32).npu()
qHeads = 32
kvHeads = 2
dimQkflat = 2048
stemBlockSize = 128
causal = True
windowSize = 4

metadata = torch.ops.npu_ops_transformer.npu_stem_indexer_metadata(
    q_seq_lens=qSeqLens,
    kv_seq_lens=kvSeqLens,
    q_heads=qHeads,
    kv_heads=kvHeads,
    causal=causal,
    dim_qkflat=dimQkflat,
    stem_block_size=stemBlockSize,
    window_size=windowSize,
)

if isinstance(metadata, torch.Tensor):
    metadata_np = metadata.cpu().numpy().flatten()
else:
    metadata_np = np.array(metadata).flatten()

section_num = int(metadata_np[0])
max_section_num = len(qSeqLens) * kvHeads
required_size = METADATA_HEADER_SIZE + max_section_num * (
    AIC_CORE_NUM * FA_METADATA_SIZE + AIV_CORE_NUM * AIV_RESERVED_METADATA_SIZE
)
expected_size = (
    (required_size + METADATA_ALIGN_SIZE - 1) // METADATA_ALIGN_SIZE
) * METADATA_ALIGN_SIZE
if len(metadata_np) != expected_size:
    raise ValueError(
        f"算子输出数据长度({len(metadata_np)})与期望长度({expected_size})不一致"
    )

fa_base = METADATA_HEADER_SIZE

print(f"section_num={section_num}, metadata_size={len(metadata_np)}")
for section_idx in range(section_num):
    print(
        f"=============================== Section {section_idx} FA ========================="
    )
    section_fa_base = fa_base + section_idx * AIC_CORE_NUM * FA_METADATA_SIZE
    for aic in range(AIC_CORE_NUM):
        offset = section_fa_base + aic * FA_METADATA_SIZE
        print(metadata_np[offset : offset + FA_METADATA_SIZE].tolist())
