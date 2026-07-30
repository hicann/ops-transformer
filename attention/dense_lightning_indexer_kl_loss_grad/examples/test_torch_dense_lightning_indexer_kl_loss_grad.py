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
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad_metadata

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

q_len = 32
compressed_k_len = 128
cmp_ratio = 4
cmp_residual_k = 0
batch_size = 1
num_heads_q = 64
num_heads_k = 1
head_dim = 128
layout = "TND"
mask_mode = 3
dtype = torch.bfloat16

q = torch.randn(q_len, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(compressed_k_len, num_heads_k, head_dim, dtype=dtype, device=device)
w = torch.randn(q_len, num_heads_q, dtype=torch.float32, device=device) * (0.1 / 6.0)
softmax_lse = torch.randn((num_heads_k, q_len), dtype=torch.float32, device=device)
attn_l1_cpu = torch.zeros((q_len, 1, compressed_k_len), dtype=torch.float32)

attn_softmax_l1_norm = attn_l1_cpu.to(device)

cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, compressed_k_len], dtype=torch.int32, device=device)
cmp_residual_k_tensor = torch.tensor([cmp_residual_k], dtype=torch.int32, device=device)

metadata = dense_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_k=compressed_k_len,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

dq, dk, dw, softmax_out = dense_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    attn_softmax_l1_norm,
    softmax_lse,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    metadata=metadata,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

torch.npu.synchronize()
print(dq.shape, dk.shape, dw.shape, softmax_out.shape)
