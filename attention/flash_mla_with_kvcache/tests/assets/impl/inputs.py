# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fill TTK auxiliary tensors in place after CSV allocation."""

import torch


def customize_inputs(
    q,
    k_cache,
    *,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    layout_kv="PA_BBND",
    mask_mode=0,
    **kwargs,
):
    for name, tensor in (
        ("cache_seqlens", cache_seqlens),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seqused_q", seqused_q),
    ):
        values = kwargs.get(f"{name}_values")
        if tensor is not None and values is not None:
            value = torch.as_tensor(values, dtype=tensor.dtype, device=tensor.device)
            if value.shape != tensor.shape:
                raise ValueError(f"{name}_values shape does not match {name}")
            tensor.copy_(value)
    if block_table is not None:
        if cache_seqlens is None:
            raise ValueError("block_table requires cache_seqlens")
        block_size = int(
            k_cache.shape[
                3 if layout_kv == "PA_NZ" else 1 if layout_kv == "PA_BBND" else 2
            ]
        )
        lengths = cache_seqlens.detach().cpu().tolist()
        if block_table.ndim != 2 or block_table.shape[0] != len(lengths):
            raise ValueError("block_table must have one row per cache sequence")
        counts = [(int(length) + block_size - 1) // block_size for length in lengths]
        if (
            any(length < 0 for length in lengths)
            or max(counts, default=0) > block_table.shape[1]
        ):
            raise ValueError("cache_seqlens exceeds block_table capacity")
        if sum(counts) > k_cache.shape[0]:
            raise ValueError("k_cache has too few physical pages")
        table = torch.full_like(block_table, -1)
        page_order = kwargs.get("page_order", "sequential")
        if page_order not in ("sequential", "reverse"):
            raise ValueError(f"Unknown page_order: {page_order}")
        page_ids = torch.arange(sum(counts), device=table.device)
        if page_order == "reverse":
            page_ids = page_ids.flip(0)
        offset = 0
        for batch, count in enumerate(counts):
            table[batch, :count] = page_ids[offset : offset + count]
            offset += count
        block_table.copy_(table)
    if attn_mask is not None:
        if mask_mode == 3:
            attn_mask.copy_(torch.ones_like(attn_mask).triu(diagonal=1))
        elif mask_mode == 0:
            attn_mask.zero_()
    if metadata is not None:
        metadata.zero_()
