#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MiniMax sparse attention split-KV torch 调用 demo（训练连续 BNSD）。

运行前：
  1. 已安装 custom opp，并设置：
       source ${ASCEND_HOME_PATH}/set_env.sh
       export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer
       export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}
  2. 已 pip install cann_ops_transformer（或 PYTHONPATH 指向 torch_extension）
  3. 首次调用会 JIT 编 C++ wrapper，需要 g++ / ninja

  python3 demo_torch_minimax_sparse_attention_split_kv.py
"""

import math
import os

import torch
import torch_npu
import cann_ops_transformer


def main():
    device_id = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    torch_npu.npu.set_device(device_id)

    # 生产约束：D=128，group_size = Nq / Nkv ∈ [1, 16]，block_size ∈ (0, 128]
    batch, s_q, s_kv = 2, 32, 256
    q_heads, kv_heads, head_dim = 64, 4, 128
    block_size, top_k = 128, 4
    scale_value = 1.0 / math.sqrt(head_dim)

    # BNSD 连续 KV，不要传 block_table
    query = torch.randn(
        batch, q_heads, s_q, head_dim, dtype=torch.bfloat16, device="npu"
    )
    key = torch.randn(
        batch, kv_heads, s_kv, head_dim, dtype=torch.bfloat16, device="npu"
    )
    value = torch.randn(
        batch, kv_heads, s_kv, head_dim, dtype=torch.bfloat16, device="npu"
    )

    actual_seq_lengths = torch.tensor([32, 24], dtype=torch.int32, device="npu")
    actual_seq_lengths_kv = torch.tensor([256, 192], dtype=torch.int32, device="npu")

    # indexer 输出：batch 内逻辑 KV block id
    # shape [Nkv, B, S, topK]，无效位填 -1
    # 真实训练把下面 fake 换成 indexer 的 select_idx 即可
    n_blocks_0 = (256 + block_size - 1) // block_size  # 2
    n_blocks_1 = (192 + block_size - 1) // block_size  # 2
    select_idx = torch.full(
        (kv_heads, batch, s_q, top_k), -1, dtype=torch.int32, device="npu"
    )
    select_idx[:, 0, :, 0] = 0
    select_idx[:, 0, :, 1] = min(1, n_blocks_0 - 1)
    select_idx[:, 1, :, 0] = 0
    select_idx[:, 1, :, 1] = min(1, n_blocks_1 - 1)
    # padding token（batch1 的 t>=24）保持 -1

    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = cann_ops_transformer.build_k2q_csr(
        select_idx,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_size,
        input_layout="BNSD",  # 会写成 padded q id = b * S + t
        # index_mode="batch_local",  # 默认；若 select_idx 是跨 batch 全局 id，改成 "global"
    )

    attn_out, softmax_lse = cann_ops_transformer.minimax_sparse_attention_split_kv(
        query,
        key,
        value,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        kv_heads,
        scale_value,
        block_size,
        top_k,
        inner_precise=4,  # 4=默认 bf16 softmax；0=fp32 softmax
        softmax_lse_flag=False,  # True 时 softmax_lse 为 [B, N, S, 1] fp32
        input_layout="BNSD",
    )
    torch.npu.synchronize()

    print("attn_out", tuple(attn_out.shape), attn_out.dtype)
    print("softmax_lse", tuple(softmax_lse.shape), softmax_lse.dtype)
    print("attn finite", torch.isfinite(attn_out.float()).all().item())


if __name__ == "__main__":
    main()
