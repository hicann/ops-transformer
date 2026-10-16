# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize metadata after H2D, before timing and graph capture."""

from .metadata import build_metadata


METADATA_SLOT_INDEX = 7


def metadata_slot_declared(tensor_dtypes):
    if tensor_dtypes is None:
        return True
    try:
        return tensor_dtypes[METADATA_SLOT_INDEX] is not None
    except (IndexError, TypeError):
        return True


def run(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    if metadata is not None and metadata.numel() != 0:
        # Concrete slot: keep the random fake tensor prepared by customize_inputs.
        return None
    if metadata is None and not metadata_slot_declared(kwargs.get("tensor_dtypes")):
        # Case file does not declare the metadata slot (dtype None): stay absent.
        return None
    generated = build_metadata(
        q,
        k_cache,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        **kwargs,
    )
    if metadata is None:
        return {"metadata": generated}
    if metadata.dtype != generated.dtype:
        raise ValueError(
            f"MLA metadata placeholder must be {generated.dtype}; got {metadata.dtype}"
        )
    metadata.resize_(generated.shape)
    metadata.copy_(generated)
