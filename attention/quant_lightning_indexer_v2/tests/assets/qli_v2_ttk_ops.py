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

"""TTK metadata-first adapter for the installed QuantLightningIndexer API."""

from typing import Optional

import torch

import cann_ops_transformer


class QuantLightningIndexerMetadataBuilder:
    """Derive and create the metadata consumed by the main operator."""

    @staticmethod
    def max_seq(cu_seqlens, seqused, fallback):
        if seqused is not None and seqused.numel() > 0:
            return int(seqused.detach().cpu().max().item())
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            values = cu_seqlens.detach().cpu()
            return int((values[1:] - values[:-1]).max().item())
        return int(fallback)

    @staticmethod
    def batch_size(query, layout_query, cu_seqlens_q, seqused_q):
        if layout_query == "BSND":
            return int(query.shape[0])
        if seqused_q is not None:
            return int(seqused_q.numel())
        if cu_seqlens_q is not None:
            return max(int(cu_seqlens_q.numel()) - 1, 0)
        return 0

    @classmethod
    def build(cls, query, key, *, cu_seqlens_q=None, cu_seqlens_k=None,
              seqused_q=None, seqused_k=None, cmp_residual_k=None,
              sparse_count=0, quant_mode=1, layout_query="BSND",
              layout_key="BSND", sparse_mode=0, cmp_ratio=1):
        q_seq = int(query.shape[1]) if layout_query == "BSND" else int(query.shape[0])
        k_seq = int(key.shape[1]) if layout_key == "BSND" else int(key.shape[0])
        q_head_num = int(query.shape[2]) if layout_query == "BSND" else int(query.shape[1])
        k_head_num = int(key.shape[1]) if layout_key == "TND" else int(key.shape[2])
        metadata = torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
            num_heads_q=q_head_num,
            num_heads_k=k_head_num,
            head_dim=int(query.shape[-1]),
            topk=int(sparse_count),
            quant_mode=int(quant_mode),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=cls.batch_size(query, layout_query, cu_seqlens_q, seqused_q),
            max_seqlen_q=cls.max_seq(cu_seqlens_q, seqused_q, q_seq),
            max_seqlen_k=cls.max_seq(cu_seqlens_k, seqused_k, k_seq),
            layout_q=layout_query,
            layout_k=layout_key,
            mask_mode=int(sparse_mode),
            cmp_ratio=int(cmp_ratio),
        )
        if hasattr(metadata, "to") and metadata.device != query.device:
            metadata = metadata.to(query.device)
        return metadata


def quant_lightning_indexer_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    sparse_count: int = 0,
    quant_mode: int = 1,
    max_seqlen_q: int = -1,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_mode: int = 0,
    cmp_ratio: int = 1,
    return_value: int = 0,
):
    metadata = QuantLightningIndexerMetadataBuilder.build(
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
    return torch.ops.cann_ops_transformer.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        int(sparse_count),
        int(quant_mode),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=int(max_seqlen_q),
        layout_q=layout_query,
        layout_k=layout_key,
        mask_mode=int(sparse_mode),
        cmp_ratio=int(cmp_ratio),
        return_value=int(return_value),
    )
