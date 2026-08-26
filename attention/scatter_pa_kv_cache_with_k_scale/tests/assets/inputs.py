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
__input__ = {
    "kernel": {"scatter_kv_cache_with_k_scale": "scatter_kv_cache_with_k_scale_input"},
    "aclnn": {
        "aclnnScatterPaKvCacheWithKScale": "aclnnScatterPaKvCacheWithKScaleInput"
    },
}

import numpy as np
import torch


def scatter_kv_cache_with_k_scale_input(
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
    input for scatter_kv_cache_with_k_scale.

    """
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[2]
    gen_num = key.shape[0]
    max_token = num_blocks * block_size
    if gen_num > max_token:
        raise ValueError("cann not gen input max_token over numBlocks * blockSize")
    slot_mapping = np.random.choice(max_token, gen_num, replace=False).astype(
        slot_mapping.dtype
    )

    return [
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
    ]


def aclnnScatterPaKvCacheWithKScaleInput(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    key_scale,
    key_scale_cache,
    cache_layout="BNBD",
    **kwargs,
):
    """
    input for aclnnScatterPaKvCacheWithKScale.

    """
    num_blocks = key_cache.shape[0]
    block_size = key_cache.shape[2]
    gen_num = key.shape[0]
    max_token = num_blocks * block_size
    if gen_num > max_token:
        raise ValueError("cann not gen input max_token over numBlocks * blockSize")
    np_type = torch.tensor([], dtype=slot_mapping.dtype).numpy().dtype
    slot_mapping = np.random.choice(max_token, gen_num, replace=False).astype(np_type)

    return [
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
    ]
