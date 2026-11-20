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

"""Fill explicitly declared FlashAttn auxiliary inputs."""

import logging
import torch


def customize_inputs(
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
    arguments = dict(locals())
    arguments.pop("kwargs")
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_kv", cu_seqlens_kv),
        ("seqused_q", seqused_q),
        ("seqused_kv", seqused_kv),
        ("sinks", sinks),
        ("seed", seed),
        ("offset", offset),
    ):
        field = f"{name}_values"
        if field in kwargs:
            if tensor is None:
                raise ValueError(f"{field} requires the {name} tensor")
            values = torch.as_tensor(
                kwargs[field], dtype=tensor.dtype, device=tensor.device
            )
            if values.shape != tensor.shape:
                raise ValueError(f"{field} shape does not match {name}")
            tensor.copy_(values)
    if "block_table_values" in kwargs:
        if block_table is None:
            raise ValueError("block_table_values requires block_table")
        values = torch.as_tensor(
            kwargs["block_table_values"],
            dtype=block_table.dtype,
            device=block_table.device,
        )
        if values.shape != block_table.shape:
            raise ValueError("block_table_values shape does not match block_table")
        block_table.copy_(values)
    elif block_table is not None:
        if layout_kv not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            raise ValueError("block_table requires a paged KV layout")
        if seqused_kv is None:
            raise ValueError("paged KV requires seqused_kv")
        bs = k.shape[3 if layout_kv == "PA_NZ" else 1 if layout_kv == "PA_BBND" else 2]
        if "block_size" in kwargs and kwargs["block_size"] != bs:
            raise ValueError("block_size does not match the KV tensor shape")
        lengths = seqused_kv.detach().cpu().tolist()
        if block_table.ndim != 2 or block_table.shape[0] != len(lengths):
            raise ValueError("block_table must have one row per KV sequence")
        counts = [(length + bs - 1) // bs for length in lengths]
        if (
            any(length < 0 for length in lengths)
            or max(counts, default=0) > block_table.shape[1]
        ):
            raise ValueError("seqused_kv exceeds block_table capacity")
        physical_pages = k.shape[0]
        if sum(counts) and physical_pages == 0:
            raise ValueError("nonempty KV sequences require physical pages")
        ids = torch.arange(sum(counts), device=block_table.device)
        if ids.numel() > physical_pages:
            logging.warning(
                "FlashAttn block_table maps %d logical pages onto %d physical pages; page IDs will repeat",
                ids.numel(),
                physical_pages,
            )
            ids = ids.remainder(physical_pages)
        order = kwargs.get("page_order", "sequential")
        if order not in ("sequential", "reverse"):
            raise ValueError(f"Unknown page_order: {order}")
        if order == "reverse":
            ids = ids.flip(0)
        block_table.fill_(-1)
        start = 0
        for batch, count in enumerate(counts):
            block_table[batch, :count] = ids[start : start + count]
            start += count
    if attn_mask is not None and mask_mode in (3, 4):
        attn_mask.copy_(torch.ones_like(attn_mask).triu(diagonal=1))
    from .cross_check import enabled, prepare

    if enabled():
        prepare(arguments, kwargs["testcase_name"])
