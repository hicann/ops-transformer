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

"""Mainline TTK graph adapter with metadata generated before capture."""

import torch


class FlashAttnAclGraph(torch.nn.Module):
    def __init__(
        self,
        q,
        k,
        v,
        *,
        block_table=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=None,
        batch_size=None,
        sinks=None,
        attn_mask=None,
        metadata=None,
        softmax_scale=1.0,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=False,
    ):
        super().__init__()
        from flash_attn_ttk_ops import build_flash_attn_metadata

        self.q = q
        self.k = k
        self.v = v
        self.block_table = block_table
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_kv = cu_seqlens_kv
        self.seqused_q = seqused_q
        self.seqused_kv = seqused_kv
        self.sinks = sinks
        self.attn_mask = attn_mask
        self.softmax_scale = float(softmax_scale)
        self.mask_mode = int(mask_mode)
        self.win_left = int(win_left)
        self.win_right = int(win_right)
        self.max_seqlen_q = int(max_seqlen_q) if max_seqlen_q is not None else -1
        self.max_seqlen_kv = int(max_seqlen_kv) if max_seqlen_kv is not None else -1
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.return_softmax_lse = bool(return_softmax_lse)
        if metadata is None:
            self.metadata = build_flash_attn_metadata(
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
        else:
            self.metadata = metadata

    def forward(self):
        return torch.ops.cann_ops_transformer.flash_attn(
            self.q,
            self.k,
            self.v,
            block_table=self.block_table,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=self.cu_seqlens_kv,
            seqused_q=self.seqused_q,
            seqused_kv=self.seqused_kv,
            sinks=self.sinks,
            attn_mask=self.attn_mask,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.mask_mode,
            win_left=self.win_left,
            win_right=self.win_right,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            layout_q=self.layout_q,
            layout_kv=self.layout_kv,
            layout_out=self.layout_out,
            return_softmax_lse=self.return_softmax_lse,
        )
