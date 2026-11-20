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

"""Reuse metadata prepared before FlashAttn graph capture."""

import torch


class FlashAttnAclGraph(torch.nn.Module):
    def __init__(
        self,
        *,
        batch_size=None,
        softmax_scale=1.0,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=False,
        dropout_p=0.0,
    ):
        super().__init__()
        self.attrs = dict(
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
            return_softmax_lse=return_softmax_lse,
            dropout_p=dropout_p,
        )

    def forward(
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
        sinks=None,
        attn_mask=None,
        metadata=None,
        seed=None,
        offset=None,
    ):
        if metadata is None or metadata.numel() == 0:
            raise ValueError(
                "FlashAttn graph requires metadata prepared by npu_preprocess"
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
            seed=seed,
            offset=offset,
            **self.attrs,
        )
