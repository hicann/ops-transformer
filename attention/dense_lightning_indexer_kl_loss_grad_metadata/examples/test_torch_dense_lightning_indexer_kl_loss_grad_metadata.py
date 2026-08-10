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
import cann_ops_transformer

metadata = torch.ops.cann_ops_transformer.dense_lightning_indexer_kl_loss_grad_metadata(
    cu_seqlens_q=torch.tensor([0, 123, 230, 234, 511], dtype=torch.int32).npu(),
    cu_seqlens_k=torch.tensor(
        [0, 3048, 4098 * 2, 4364 * 3, 4098 * 4], dtype=torch.int32
    ).npu(),
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=4,
    max_seqlen_q=180,
    max_seqlen_k=5,
    num_heads_q=64,
    num_heads_k=1,
    head_dim=128,
    mask_mode=0,
    layout_q="TND",
    layout_k="TND",
    cmp_ratio=13,
)
