#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import random
import numpy as np


def do_golden(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    block_size,
    num_heads,
    head_size_k,
    head_size_v,
    last_dim_k,
    last_dim_v,
):
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block_index = slot // block_size
        block_offset = slot % block_size

        token_key = key[i].reshape(num_heads * head_size_k)
        for k in range(num_heads * head_size_k // last_dim_k):
            key_cache[block_index][k][block_offset][:] = token_key[
                k * last_dim_k : k * last_dim_k + last_dim_k
            ]

        token_value = value[i].reshape(num_heads * head_size_v)
        for v in range(num_heads * head_size_v // last_dim_v):
            value_cache[block_index][v][block_offset][:] = token_value[
                v * last_dim_v : v * last_dim_v + last_dim_v
            ]

    return key_cache, value_cache


def get_dtype_size(dtype):
    if dtype == "int8":
        return 1
    else:
        return 2


def gen_golden_data_simple(params):
    num_tokens = params["num_tokens"]
    num_heads = params["num_heads"]
    head_size_k = params["head_size_k"]
    head_size_v = params["head_size_v"]
    block_size = params["block_size"]
    num_blocks = params["num_blocks"]
    dtype_k = params["k_type"]
    dtype_v = params["v_type"]
    dtype_slot_mapping = params["slot_mapping_type"]
    last_dim_k = 32 // get_dtype_size(dtype_k)
    last_dim_v = 32 // get_dtype_size(dtype_v)
    key = np.random.uniform(
        -5.0, 5.0, size=(num_tokens, num_heads, head_size_k)
    ).astype(dtype_k)
    value = np.random.uniform(
        -5.0, 5.0, size=(num_tokens, num_heads, head_size_v)
    ).astype(dtype_v)
    num_slots = block_size * num_blocks
    slot_mapping = np.array(random.sample(range(0, num_slots), num_tokens)).astype(
        dtype_slot_mapping
    )
    key_cache = np.zeros(
        (num_blocks, num_heads * head_size_k // last_dim_k, block_size, last_dim_k)
    ).astype(dtype_k)
    value_cache = np.zeros(
        (num_blocks, num_heads * head_size_v // last_dim_v, block_size, last_dim_v)
    ).astype(dtype_v)
    key_cache.tofile("./input_key_cache.bin")
    value_cache.tofile("./input_value_cache.bin")

    key_cache, value_cache = do_golden(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        block_size,
        num_heads,
        head_size_k,
        head_size_v,
        last_dim_k,
        last_dim_v,
    )

    key.tofile("./input_key.bin")
    value.tofile("./input_value.bin")
    slot_mapping.tofile("./input_slot_mapping.bin")
    key_cache.tofile("./golden_key_cache.bin")
    value_cache.tofile("./golden_value_cache.bin")


params_info = {
    "test_scatter_pa_kv_cache_full_load1": {
        "num_tokens": 256,
        "num_heads": 1,
        "head_size_k": 512,
        "head_size_v": 64,
        "block_size": 16,
        "num_blocks": 20,
        "k_type": "float16",
        "v_type": "float16",
        "slot_mapping_type": "int32",
    },
    "test_scatter_pa_kv_cache_full_load2": {
        "num_tokens": 256,
        "num_heads": 1,
        "head_size_k": 512,
        "head_size_v": 64,
        "block_size": 16,
        "num_blocks": 20,
        "k_type": "float16",
        "v_type": "float16",
        "slot_mapping_type": "int64",
    },
    "test_case_nz_not_full_load_001": {
        "num_tokens": 256,
        "num_heads": 1,
        "head_size_k": 512,
        "head_size_v": 64,
        "block_size": 16,
        "num_blocks": 20,
        "k_type": "float16",
        "v_type": "float16",
        "slot_mapping_type": "int32",
    },
    "test_case_nz_not_full_load_002": {
        "num_tokens": 256,
        "num_heads": 1,
        "head_size_k": 512,
        "head_size_v": 64,
        "block_size": 16,
        "num_blocks": 20,
        "k_type": "float16",
        "v_type": "float16",
        "slot_mapping_type": "int64",
    },
}

if __name__ == "__main__":
    params_list = params_info[sys.argv[1]]
    gen_golden_data_simple(params_list)
