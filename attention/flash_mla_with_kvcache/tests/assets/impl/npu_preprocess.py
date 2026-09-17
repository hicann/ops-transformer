# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize the metadata placeholder after H2D and before timing."""

from .metadata import build_metadata


def run(
    q,
    k_cache,
    *,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    **kwargs,
):
    if metadata is None:
        # The metadata-first wrapper creates it when no slot was allocated.
        return None
    generated = build_metadata(
        q,
        k_cache,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        **kwargs,
    )
    if metadata.dtype != generated.dtype or metadata.numel() < generated.numel():
        raise ValueError(
            f"MLA metadata placeholder must be {generated.dtype} with at least "
            f"{generated.numel()} elements; got {metadata.dtype}, {metadata.numel()}"
        )
    metadata.zero_()
    metadata.reshape(-1)[: generated.numel()].copy_(generated.reshape(-1))
