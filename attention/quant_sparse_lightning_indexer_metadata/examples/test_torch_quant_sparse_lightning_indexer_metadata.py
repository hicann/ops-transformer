# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import cann_ops_transformer as cot

torch.npu.set_device(0)
device = "npu:0"
batch, key_length = 2, 2048
# 每个batch包含一个Query，每行有256个有效候选逻辑块，块大小为8。
lengths = torch.full((batch, 1), 256, dtype=torch.int32, device=device)
cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
used_k = torch.full((batch,), key_length, dtype=torch.int32, device=device)
metadata = cot.quant_sparse_lightning_indexer_metadata(
    32,
    1,
    128,
    512,
    5,
    cu_seqlens_q=cu_q,
    seqused_k=used_k,
    batch_size=batch,
    max_seqlen_q=1,
    max_seqlen_k=key_length,
    candidate_block_length=lengths,
    candidate_block_size=8,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=0,
    cmp_ratio=1,
)
torch.npu.synchronize()
print(metadata.shape)  # (1024,)
