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

"""Build dynamic metadata after H2D and before timing or graph capture."""

import torch

METADATA_INDEX = 10


def metadata_slot_declared(tensor_dtypes):
    if (
        not isinstance(tensor_dtypes, (tuple, list))
        or not 11 <= len(tensor_dtypes) <= 13
    ):
        raise ValueError(
            "tensor_dtypes must include the FlashAttn metadata slot and optional seed/offset slots"
        )
    return tensor_dtypes[METADATA_INDEX] is not None


def build_metadata_arguments(q, k, v, kwargs):
    layout_q, layout_kv = kwargs["layout_q"], kwargs["layout_kv"]
    if layout_q not in ("BSND", "BNSD", "TND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_kv not in ("BSND", "BNSD", "TND", "PA_BBND", "PA_BNBD", "PA_NZ"):
        raise ValueError(f"unsupported layout_kv: {layout_kv}")
    batch_size = kwargs.get("batch_size")
    if batch_size is None and layout_q != "TND":
        batch_size = q.shape[0]
    return dict(
        num_heads_q=int(q.shape[1 if layout_q in ("TND", "BNSD") else 2]),
        num_heads_kv=int(
            k.shape[1 if layout_kv in ("TND", "BNSD", "PA_BNBD", "PA_NZ") else 2]
        ),
        head_dim=int(q.shape[-1]),
        head_dim_v=int(
            v.shape[2] * v.shape[4] if layout_kv == "PA_NZ" else v.shape[-1]
        ),
        cu_seqlens_q=kwargs["cu_seqlens_q"],
        cu_seqlens_kv=kwargs["cu_seqlens_kv"],
        seqused_q=kwargs["seqused_q"],
        seqused_kv=kwargs["seqused_kv"],
        batch_size=batch_size,
        max_seqlen_q=kwargs["max_seqlen_q"],
        max_seqlen_kv=kwargs["max_seqlen_kv"],
        mask_mode=kwargs["mask_mode"],
        win_left=kwargs["win_left"],
        win_right=kwargs["win_right"],
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=kwargs["layout_out"],
        dropout_p=kwargs.get("dropout_p", 0.0),
    )


def run(
    q,
    k,
    v,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
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
    seed=None,
    offset=None,
    **kwargs,
):
    if metadata is not None:
        return None
    if not metadata_slot_declared(kwargs["tensor_dtypes"]):
        return None
    import cann_ops_transformer  # noqa: F401

    arguments = dict(locals())
    arguments["batch_size"] = kwargs.get("batch_size")
    generated = torch.ops.cann_ops_transformer.flash_attn_metadata(
        **build_metadata_arguments(q, k, v, arguments)
    )
    return {"metadata": generated.to(device=q.device)}
