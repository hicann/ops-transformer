#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np

__golden__ = {
    "kernel": {"scatter_kv_cache_with_k_scale": "scatter_kv_cache_with_k_scale_golden"},
    "aclnn": {
        "aclnnScatterPaKvCacheWithKScale": "aclnnScatterPaKvCacheWithKScaleGolden"
    },
}


def scatter_kv_cache_with_k_scale_golden(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    key_scale,
    key_scale_cache,
    *,
    cache_layout="BNBD",
    **kwargs,
):
    """
    Kernel golden for scatter_kv_cache_with_k_scale.

    Pure memory-access operator: scatter key/value tensors into KV cache
    according to slot_mapping indices, and write key_scale values into
    key_scale_cache. Inplace operator.

    Parameters follow SE document definition order:
    - key:             [num_tokens, num_head, k_head_size]         FP8 (uint8 storage)
    - value:           [num_tokens, num_head, v_head_size]         FP8 (uint8 storage)
    - key_cache:       [num_blocks, num_head, block_size, k_head_size]  FP8 (inplace, uint8 storage)
    - value_cache:     [num_blocks, num_head, block_size, v_head_size]  FP8 (inplace, uint8 storage)
    - slot_mapping:    [num_tokens]                                int32/int64
    - key_scale:       [num_tokens, num_head]                      float32
    - key_scale_cache: [num_blocks, num_head, block_size, 1]       float32 (inplace)
    - cache_layout:    str, default "BNBD"

    Returns:
        tuple: (key_cache, value_cache, key_scale_cache) - inplace-modified inputs
    """
    num_tokens = key.shape[0]
    num_head = key.shape[1]
    k_head_size = key.shape[2]
    v_head_size = value.shape[2]
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[2]

    # Input shape validation (defensive)
    assert key.shape == (num_tokens, num_head, k_head_size), (
        f"key shape mismatch: expected ({num_tokens}, {num_head}, {k_head_size}), got {key.shape}"
    )
    assert value.shape == (num_tokens, num_head, v_head_size), (
        f"value shape mismatch: expected ({num_tokens}, {num_head}, {v_head_size}), got {value.shape}"
    )
    assert key_cache.shape == (num_blocks, num_head, block_size, k_head_size), (
        f"key_cache shape mismatch: expected ({num_blocks}, {num_head}, {block_size}, {k_head_size}), got {key_cache.shape}"
    )
    assert value_cache.shape == (num_blocks, num_head, block_size, v_head_size), (
        f"value_cache shape mismatch: expected ({num_blocks}, {num_head}, {block_size}, {v_head_size}), got {value_cache.shape}"
    )
    assert slot_mapping.shape == (num_tokens,), (
        f"slot_mapping shape mismatch: expected ({num_tokens},), got {slot_mapping.shape}"
    )
    assert key_scale.shape == (num_tokens, num_head), (
        f"key_scale shape mismatch: expected ({num_tokens}, {num_head}), got {key_scale.shape}"
    )
    assert key_scale_cache.shape == (num_blocks, num_head, block_size, 1), (
        f"key_scale_cache shape mismatch: expected ({num_blocks}, {num_head}, {block_size}, 1), got {key_scale_cache.shape}"
    )

    max_slot = num_blocks * block_size
    for i in range(num_tokens):
        slot = int(slot_mapping[i])
        if slot < 0 or slot >= max_slot:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        for j in range(num_head):
            key_cache[block_idx, j, block_offset, :] = key[i, j, :]
            value_cache[block_idx, j, block_offset, :] = value[i, j, :]
            key_scale_cache[block_idx, j, block_offset, 0] = key_scale[i, j]
    return key_cache, value_cache, key_scale_cache


def aclnnScatterPaKvCacheWithKScaleGolden(
    key,
    value,
    key_cache_ref,
    value_cache_ref,
    slot_mapping,
    key_scale,
    key_scale_cache_ref,
    cache_layout="BNBD",
    **kwargs,
):
    num_tokens = key.shape[0]
    num_head = key.shape[1]
    k_head_size = key.shape[2]
    v_head_size = value.shape[2]
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[2]

    max_slot = num_blocks * block_size

    for i in range(num_tokens):
        slot = int(slot_mapping[i])
        if slot < 0 or slot >= max_slot:
            continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        for j in range(num_head):
            # scatter key (raw memory copy, dtype-agnostic)
            key_cache_ref[block_idx, j, block_offset, :] = key[i, j, :]
            value_cache_ref[block_idx, j, block_offset, :] = value[i, j, :]
            key_scale_cache_ref[block_idx, j, block_offset, 0] = key_scale[i, j]
