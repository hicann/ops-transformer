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

"""ACLGraph adapter with LightningIndexer metadata generated before capture."""

import torch


class LightningIndexerV2AclGraph(torch.nn.Module):
    def __init__(
        self,
        q,
        k,
        w,
        *,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        cmp_residual_k=None,
        block_table=None,
        output_idx_offset=None,
        topk=0,
        max_seqlen_q=-1,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=0,
        cmp_ratio=1,
        return_value=0,
    ):
        super().__init__()
        from li_v2_ttk_ops import LightningIndexerMetadataBuilder

        self.q = q
        self.k = k
        self.w = w
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.seqused_q = seqused_q
        self.seqused_k = seqused_k
        self.cmp_residual_k = cmp_residual_k
        self.block_table = block_table
        self.output_idx_offset = output_idx_offset
        self.topk = int(topk)
        self.max_seqlen_q = int(max_seqlen_q)
        self.layout_q = layout_q
        self.layout_k = layout_k
        self.mask_mode = int(mask_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.return_value = int(return_value)
        self.metadata = LightningIndexerMetadataBuilder.build(
            q,
            k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            topk=topk,
            layout_q=layout_q,
            layout_k=layout_k,
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
        )

    def forward(self):
        return torch.ops.cann_ops_transformer.lightning_indexer(
            self.q,
            self.k,
            self.w,
            self.topk,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_k,
            seqused_q=self.seqused_q,
            seqused_k=self.seqused_k,
            cmp_residual_k=self.cmp_residual_k,
            block_table=self.block_table,
            output_idx_offset=self.output_idx_offset,
            metadata=self.metadata,
            max_seqlen_q=self.max_seqlen_q,
            layout_q=self.layout_q,
            layout_k=self.layout_k,
            mask_mode=self.mask_mode,
            cmp_ratio=self.cmp_ratio,
            return_value=self.return_value,
        )
