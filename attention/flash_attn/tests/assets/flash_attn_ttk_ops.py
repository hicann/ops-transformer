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

"""TTK metadata-first adapter for the installed FlashAttn API."""

from typing import Optional

import torch

import cann_ops_transformer


class FlashAttnMetadataBuilder:
    """Derive scalar metadata parameters from operator inputs.

    Rule: only use explicitly provided values; do not auto-generate missing
    parameters. The flash_attn_metadata op handles -1/None internally.
    """

    @staticmethod
    def resolve_max_seqlen(explicit):
        if explicit is not None and int(explicit) > 0:
            return int(explicit)
        return -1


def build_flash_attn_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    **_unused,
):
    """Build metadata before the main op or graph capture."""
    head_dim = int(q.shape[-1])

    if layout_q == "TND":
        num_heads_q = int(q.shape[1])
    elif layout_q == "BNSD":
        num_heads_q = int(q.shape[1])
    else:
        num_heads_q = int(q.shape[2])

    if layout_kv == "TND":
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "BNSD":
        num_heads_kv = int(k.shape[1])
    elif layout_kv in ("PA_BNBD", "PA_NZ"):
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "PA_BBND":
        num_heads_kv = int(k.shape[2])
    else:
        num_heads_kv = int(k.shape[2])

    msq = FlashAttnMetadataBuilder.resolve_max_seqlen(max_seqlen_q)
    msk = FlashAttnMetadataBuilder.resolve_max_seqlen(max_seqlen_kv)

    metadata = torch.ops.cann_ops_transformer.flash_attn_metadata(
        int(num_heads_q),
        int(num_heads_kv),
        int(head_dim),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=int(msq),
        max_seqlen_kv=int(msk),
        mask_mode=int(mask_mode),
        win_left=int(win_left),
        win_right=int(win_right),
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
    )
    if hasattr(metadata, "to") and metadata.device != q.device:
        metadata = metadata.to(q.device)
    return metadata


def flash_attn_ttk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
):
    """Generate metadata in TTK, then call the flash_attn extension op.

    If metadata is provided, use it directly; otherwise build it via
    build_flash_attn_metadata.
    """
    if metadata is None:
        metadata = build_flash_attn_metadata(
            q,
            k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )

    return torch.ops.cann_ops_transformer.flash_attn(
        q,
        k,
        v,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        attn_mask=attn_mask,
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        mask_mode=int(mask_mode),
        win_left=int(win_left),
        win_right=int(win_right),
        max_seqlen_q=int(max_seqlen_q) if max_seqlen_q is not None else -1,
        max_seqlen_kv=int(max_seqlen_kv) if max_seqlen_kv is not None else -1,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        return_softmax_lse=bool(return_softmax_lse),
    )
