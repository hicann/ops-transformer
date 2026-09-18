# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resize and fill metadata after H2D, before timing and graph capture."""

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
    if metadata.dtype != generated.dtype:
        raise ValueError(
            f"MLA metadata placeholder must be {generated.dtype}; got {metadata.dtype}"
        )
    metadata.resize_(generated.shape)
    metadata.copy_(generated)
