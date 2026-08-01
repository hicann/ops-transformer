#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK metadata-first adapter for the installed quant_block_sparse_attn API."""

import math
import sys
from pathlib import Path
from typing import Optional

import torch

_CUSTOM_OPS_LOADED = False


def _ensure_cann_ops():
    global _CUSTOM_OPS_LOADED
    if not _CUSTOM_OPS_LOADED:
        pytest_dir = str(Path(__file__).resolve().parents[1] / "pytest")
        if pytest_dir not in sys.path:
            sys.path.insert(0, pytest_dir)
        import custom_ops  # noqa: F401 - loads .so + registers torch.ops.custom namespace

        _CUSTOM_OPS_LOADED = True


_FP32_BYTES = 4


def _pack_combined_kv(key, value, k_descale):
    """Pack separate PA key/value/k_descale into a combined KV storage.

    Layout per physical block: [key_segment (FP8)] [value_segment (FP8)] [k_scale_segment (FP32)]
    Returns 4D PA views whose shape matches the operator interface and whose block stride
    points at the full physical KV-cache block.
    """
    num_blocks, n2, block_size, d = (
        int(key.shape[0]),
        int(key.shape[1]),
        int(key.shape[2]),
        int(key.shape[3]),
    )
    dv = int(value.shape[-1])

    key_seg = n2 * block_size * d
    value_seg = n2 * block_size * dv
    k_scale_seg = n2 * block_size * _FP32_BYTES
    pa_block_stride = key_seg + value_seg + k_scale_seg

    storage = torch.zeros(
        (num_blocks * pa_block_stride,), dtype=torch.uint8, device=key.device
    )
    fp8_storage = storage.view(torch.float8_e4m3fn)
    fp32_storage = storage.view(torch.float32)

    key_view = torch.as_strided(
        fp8_storage, key.shape, (pa_block_stride, block_size * d, d, 1), 0
    )
    value_view = torch.as_strided(
        fp8_storage, value.shape, (pa_block_stride, block_size * dv, dv, 1), key_seg
    )
    k_scale_shape = tuple(k_descale.shape) if k_descale.dim() == 4 else tuple(k_descale.shape) + (1,)
    k_scale_view = torch.as_strided(
        fp32_storage,
        k_scale_shape,
        (pa_block_stride // _FP32_BYTES, block_size, 1, 1),
        (key_seg + value_seg) // _FP32_BYTES,
    )

    key_view.copy_(key)
    value_view.copy_(value)
    k_scale_view.copy_(k_descale if k_descale.dim() == 4 else k_descale.unsqueeze(-1))

    return key_view, value_view, k_scale_view


def _auto_generate_params(
    query,
    key,
    sparse_indices,
    seqused_q,
    seqused_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table,
    max_seqlen_q,
    max_seqlen_kv,
    layout_q,
    sparse_kv_block_size,
):
    """Auto-generate missing seqused/cu_seqlens/block_table from available info."""
    B = int(sparse_indices.shape[0]) if sparse_indices is not None else 1

    if max_seqlen_q <= 0:
        if layout_q == "BSND":
            max_seqlen_q = int(query.shape[1])
        elif layout_q == "NTD":
            max_seqlen_q = int(query.shape[1])
        else:
            max_seqlen_q = int(query.shape[0])

    if max_seqlen_kv <= 0:
        if key.dim() == 4:
            max_seqlen_kv = int(key.shape[0]) * int(key.shape[2])
        else:
            max_seqlen_kv = int(key.shape[0])

    if seqused_q is None:
        seqused_q = torch.full(
            (B,), max_seqlen_q, dtype=torch.int32, device=query.device
        )
    if seqused_kv is None:
        seqused_kv = torch.full(
            (B,), max_seqlen_kv, dtype=torch.int32, device=query.device
        )
    if cu_seqlens_q is None:
        cu = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
        for i in range(B):
            cu[i + 1] = cu[i] + int(seqused_q[i].item())
        cu_seqlens_q = cu
    if cu_seqlens_kv is None:
        cu = torch.zeros(B + 1, dtype=torch.int32, device=query.device)
        for i in range(B):
            cu[i + 1] = cu[i] + int(seqused_kv[i].item())
        cu_seqlens_kv = cu
    if block_table is None:
        num_blocks = math.ceil(max_seqlen_kv / sparse_kv_block_size)
        block_table = torch.arange(
            B * num_blocks, dtype=torch.int32, device=query.device
        ).reshape(B, num_blocks)

    return seqused_q, seqused_kv, cu_seqlens_q, cu_seqlens_kv, block_table


class QuantBlockSparseAttnMetadataBuilder:
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
    def batch_size(q, layout_q, cu_seqlens_q, seqused_q, sparse_indices=None):
        if layout_q == "BSND":
            return int(q.shape[0])
        if seqused_q is not None:
            return int(seqused_q.numel())
        if cu_seqlens_q is not None:
            return max(int(cu_seqlens_q.numel()) - 1, 0)
        if sparse_indices is not None:
            return int(sparse_indices.shape[0])
        return 1

    @classmethod
    def build(
        cls,
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=None,
        block_table=None,
        softmax_scale=1.0,
        sparse_q_block_size=128,
        sparse_kv_block_size=128,
        layout_q="BSND",
        layout_kv="PA_BNSD",
        layout_sparse_indices="B_N_Qb_Kb",
        mask_mode=3,
        quant_mode=1,
        max_seqlen_q=0,
        max_seqlen_kv=0,
    ):
        _ensure_cann_ops()
        if layout_q == "BSND":
            num_heads_q = int(query.shape[2])
            q_fallback = int(query.shape[1])
        elif layout_q == "NTD":
            num_heads_q = int(query.shape[0])
            q_fallback = int(query.shape[1])
        else:
            num_heads_q = int(query.shape[1])
            q_fallback = int(query.shape[0])
        num_heads_kv = int(key.shape[1])
        kv_fallback = (
            int(key.shape[0] * key.shape[2]) if key.dim() == 4 else int(key.shape[0])
        )
        metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
            sparse_seq_len,
            num_heads_q,
            num_heads_kv,
            int(query.shape[-1]),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=cls.batch_size(
                query, layout_q, cu_seqlens_q, seqused_q, sparse_indices
            ),
            sparse_block_size_q=int(sparse_q_block_size),
            sparse_block_size_k=int(sparse_kv_block_size),
            quant_mode=int(quant_mode),
            mask_mode=int(mask_mode),
            max_seqlen_q=cls.max_seq(
                cu_seqlens_q, seqused_q, max_seqlen_q or q_fallback
            ),
            max_seqlen_kv=cls.max_seq(
                cu_seqlens_kv, seqused_kv, max_seqlen_kv or kv_fallback
            ),
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
        )
        if hasattr(metadata, "to") and metadata.device != query.device:
            metadata = metadata.to(query.device)
        return metadata


def quant_block_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    p_scale: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_seq_len: torch.Tensor,
    atten_mask: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    sparse_q_block_size: int = 128,
    sparse_kv_block_size: int = 128,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    layout_kv: str = "PA_BNSD",
    layout_q: str = "BSND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    layout_out: str = "TND",
    quant_mode: int = 1,
    mask_mode: int = 3,
    return_softmax_lse: bool = False,
):
    """Compute metadata first, then call cann_ops_transformer.quant_block_sparse_attn."""
    _ensure_cann_ops()

    if seqused_q is None and cu_seqlens_q is None:
        seqused_q, seqused_kv, cu_seqlens_q, cu_seqlens_kv, block_table = (
            _auto_generate_params(
                query,
                key,
                sparse_indices,
                seqused_q,
                seqused_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                block_table,
                max_seqlen_q,
                max_seqlen_kv,
                layout_q,
                sparse_kv_block_size,
            )
        )

    if metadata is None or metadata.numel() < 584:
        metadata = QuantBlockSparseAttnMetadataBuilder.build(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            block_table=block_table,
            softmax_scale=softmax_scale,
            sparse_q_block_size=sparse_q_block_size,
            sparse_kv_block_size=sparse_kv_block_size,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            mask_mode=mask_mode,
            quant_mode=quant_mode,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
        )

    if key.dim() == 4 and int(quant_mode) != 2:
        key, value, k_descale = _pack_combined_kv(key, value, k_descale)
    return torch.ops.custom.npu_quant_block_sparse_attn(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        sparse_q_block_size=int(sparse_q_block_size),
        sparse_kv_block_size=int(sparse_kv_block_size),
        layout_kv=layout_kv,
        layout_q=layout_q,
        layout_sparse_indices=layout_sparse_indices,
        layout_out=layout_out,
        quant_mode=int(quant_mode),
        mask_mode=int(mask_mode),
        return_softmax_lse=bool(return_softmax_lse),
    )
