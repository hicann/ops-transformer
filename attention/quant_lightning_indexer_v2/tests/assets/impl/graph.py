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

"""ACLGraph adapter with QuantLightningIndexer metadata generated before capture."""

import torch


class QuantLightningIndexerV2AclGraph(torch.nn.Module):
    def __init__(
        self,
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        *,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        cmp_residual_k=None,
        block_table=None,
        output_idx_offset=None,
        sparse_count=0,
        quant_mode=1,
        max_seqlen_q=-1,
        layout_query="BSND",
        layout_key="BSND",
        sparse_mode=0,
        cmp_ratio=1,
        return_value=0,
    ):
        super().__init__()
        from qli_v2_ttk_ops import QuantLightningIndexerMetadataBuilder

        self.query = query
        self.key = key
        self.weights = weights
        self.query_dequant_scale = query_dequant_scale
        self.key_dequant_scale = key_dequant_scale
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.seqused_q = seqused_q
        self.seqused_k = seqused_k
        self.cmp_residual_k = cmp_residual_k
        self.block_table = block_table
        self.output_idx_offset = output_idx_offset
        self.sparse_count = int(sparse_count)
        self.quant_mode = int(quant_mode)
        self.max_seqlen_q = int(max_seqlen_q)
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_mode = int(sparse_mode)
        self.cmp_ratio = int(cmp_ratio)
        self.return_value = int(return_value)
        self.metadata = QuantLightningIndexerMetadataBuilder.build(
            query,
            key,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            sparse_count=sparse_count,
            quant_mode=quant_mode,
            layout_query=layout_query,
            layout_key=layout_key,
            sparse_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )

    def forward(self):
        return torch.ops.cann_ops_transformer.quant_lightning_indexer(
            self.query,
            self.key,
            self.weights,
            self.query_dequant_scale,
            self.key_dequant_scale,
            self.sparse_count,
            self.quant_mode,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_k=self.cu_seqlens_k,
            seqused_q=self.seqused_q,
            seqused_k=self.seqused_k,
            cmp_residual_k=self.cmp_residual_k,
            block_table=self.block_table,
            output_idx_offset=self.output_idx_offset,
            metadata=self.metadata,
            max_seqlen_q=self.max_seqlen_q,
            layout_q=self.layout_query,
            layout_k=self.layout_key,
            mask_mode=self.sparse_mode,
            cmp_ratio=self.cmp_ratio,
            return_value=self.return_value,
        )
