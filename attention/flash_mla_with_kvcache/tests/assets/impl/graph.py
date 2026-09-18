# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reuse MLA metadata prepared before graph capture."""

import torch


class FlashMlaWithKvcacheAclGraph(torch.nn.Module):
    def __init__(
        self,
        *,
        head_dim_v=512,
        softmax_scale=1.0,
        mask_mode=0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="BSND",
        layout_kv="PA_BBND",
        layout_out="BSND",
        return_softmax_lse=False,
    ):
        super().__init__()
        self.attrs = dict(
            head_dim_v=head_dim_v,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=return_softmax_lse,
        )

    def forward(
        self,
        q,
        k_cache,
        *,
        block_table=None,
        cache_seqlens=None,
        cu_seqlens_q=None,
        seqused_q=None,
        attn_mask=None,
        metadata=None,
    ):
        tensors = dict(
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            attn_mask=attn_mask,
        )
        if metadata is None or metadata.numel() == 0:
            raise ValueError("MLA graph requires metadata prepared by npu_preprocess")
        return torch.ops.cann_ops_transformer.flash_mla_with_kvcache(
            q, k_cache, **tensors, metadata=metadata, **self.attrs
        )
