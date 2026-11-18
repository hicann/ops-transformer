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
import cann_ops_transformer as cot

torch.npu.set_device(0)
device = "npu:0"
batch, heads, head_dim = 2, 32, 128
key_length, page_size, candidate_size = 2048, 128, 8
pages_per_batch = key_length // page_size
num_pages = batch * pages_per_batch

# 构造量化输入与独立的物理页映射。
q = torch.randint(0, 256, (batch, heads, head_dim // 2), dtype=torch.uint8).to(device)
key_data = torch.randint(
    0, 256, (num_pages, page_size, 1, head_dim // 2), dtype=torch.uint8
).to(device)
key_scale = torch.full(
    (num_pages, page_size, 1, 2, 2), 127, dtype=torch.uint8, device=device
)
# PA_BBND每8个token合并存储512字节Key和32字节scale。
k = torch.cat(
    (
        key_data.reshape(num_pages, page_size // 8, 512),
        key_scale.reshape(num_pages, page_size // 8, 32),
    ),
    dim=-1,
)
w = torch.ones((batch, heads), dtype=torch.float32, device=device)
scale_q = torch.full((batch, heads, 2, 2), 127, dtype=torch.uint8, device=device).view(
    torch.float8_e8m0fnu
)
scale_k = torch.empty((0,), dtype=torch.uint8, device=device)
cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
used_k = torch.full((batch,), key_length, dtype=torch.int32, device=device)
block_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
    batch, pages_per_batch
)

# 候选容量固定为2048，只有前256个逻辑块有效，其余位置填-1。
valid_candidates = key_length // candidate_size
candidates = torch.full((batch, 1, 2048), -1, dtype=torch.int32, device=device)
candidates[:, 0, :valid_candidates] = torch.arange(
    valid_candidates, dtype=torch.int32, device=device
)
lengths = torch.full((batch, 1), valid_candidates, dtype=torch.int32, device=device)
attrs = dict(
    topk=512,
    quant_mode=5,
    candidate_block_size=candidate_size,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=0,
    cmp_ratio=1,
    max_seqlen_q=1,
)

# 使用本次候选长度及序列信息生成metadata，再执行计算。
metadata = cot.quant_sparse_lightning_indexer_metadata(
    num_heads_q=heads,
    num_heads_k=1,
    head_dim=head_dim,
    cu_seqlens_q=cu_q,
    seqused_k=used_k,
    batch_size=batch,
    max_seqlen_k=key_length,
    candidate_block_length=lengths,
    **attrs,
)
sparse_indices, sparse_values = cot.quant_sparse_lightning_indexer(
    q,
    k,
    w,
    scale_q,
    scale_k,
    candidate_block_indices=candidates,
    candidate_block_length=lengths,
    cu_seqlens_q=cu_q,
    seqused_k=used_k,
    block_table=block_table,
    metadata=metadata,
    return_value=0,
    **attrs,
)
torch.npu.synchronize()
print(sparse_indices.shape)  # (2, 1, 512)
print(sparse_values.shape)  # (0,)
