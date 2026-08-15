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

import torch


FP32_BYTES = 4
PACKING = "block_segmented_k_v_k_descale"


def _fp8_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for combined KV cache views"
        )
    return torch.float8_e4m3fn


def make_combined_kv_meta(
    block_num, block_size, n2, d_size, dv_size, block_padding_bytes, layout_kv
):
    if layout_kv != "PA_BNBD":
        raise ValueError(f"unsupported combined KV layout: {layout_kv}")
    if block_padding_bytes != 0:
        raise ValueError("segmented combined KV layout does not support block padding")

    key_segment_bytes = n2 * block_size * d_size
    value_segment_bytes = n2 * block_size * dv_size
    k_scale_segment_bytes = n2 * block_size * FP32_BYTES
    pa_block_stride = key_segment_bytes + value_segment_bytes + k_scale_segment_bytes
    k_scale_offset_bytes = key_segment_bytes + value_segment_bytes
    if k_scale_offset_bytes % FP32_BYTES != 0:
        raise ValueError(
            f"k_scale offset should be divisible by {FP32_BYTES}, got {k_scale_offset_bytes}"
        )
    if pa_block_stride % FP32_BYTES != 0:
        raise ValueError(
            f"pa_block_stride should be divisible by {FP32_BYTES}, got {pa_block_stride}"
        )

    return {
        "packing": PACKING,
        "layout_kv": layout_kv,
        "block_num": block_num,
        "block_size": block_size,
        "n2": n2,
        "d_size": d_size,
        "dv_size": dv_size,
        "block_padding_bytes": 0,
        "key_segment_bytes": key_segment_bytes,
        "value_segment_bytes": value_segment_bytes,
        "k_scale_segment_bytes": k_scale_segment_bytes,
        "pa_block_stride": pa_block_stride,
        "key_offset_bytes": 0,
        "value_offset_bytes": key_segment_bytes,
        "k_scale_offset_bytes": k_scale_offset_bytes,
    }


def make_combined_kv_views(storage, meta):
    if (
        storage.dtype != torch.uint8
        or storage.dim() != 1
        or not storage.is_contiguous()
    ):
        raise ValueError(
            "combined KV storage should be a contiguous rank-1 uint8 tensor"
        )

    block_num = meta["block_num"]
    block_size = meta["block_size"]
    n2 = meta["n2"]
    d_size = meta["d_size"]
    dv_size = meta["dv_size"]
    pa_block_stride = meta["pa_block_stride"]
    required_bytes = block_num * pa_block_stride
    if storage.numel() < required_bytes:
        raise ValueError(
            f"combined KV storage has {storage.numel()} bytes, expected at least {required_bytes}"
        )
    if storage.numel() % FP32_BYTES != 0:
        raise ValueError(
            f"combined KV storage byte length should be divisible by {FP32_BYTES}"
        )

    if meta["layout_kv"] != "PA_BNBD":
        raise ValueError(f"unsupported combined KV layout: {meta['layout_kv']}")
    if meta.get("packing") != PACKING:
        raise ValueError(f"unsupported combined KV packing: {meta.get('packing')}")

    fp8_storage = storage.view(_fp8_dtype())
    key_stride = (pa_block_stride, block_size * d_size, d_size, 1)
    value_stride = (pa_block_stride, block_size * dv_size, dv_size, 1)
    key = torch.as_strided(
        fp8_storage,
        (block_num, n2, block_size, d_size),
        key_stride,
        meta["key_offset_bytes"],
    )
    value = torch.as_strided(
        fp8_storage,
        (block_num, n2, block_size, dv_size),
        value_stride,
        meta["value_offset_bytes"],
    )

    fp32_storage = storage.view(torch.float32)
    k_scale_stride = (pa_block_stride // FP32_BYTES, block_size, 1, 1)
    k_scale = torch.as_strided(
        fp32_storage,
        (block_num, n2, block_size, 1),
        k_scale_stride,
        meta["k_scale_offset_bytes"] // FP32_BYTES,
    )
    return key, value, k_scale


def pack_combined_kv_cache(
    dense_key,
    dense_value,
    dense_k_scale,
    seqused_kv,
    block_table,
    block_size,
    block_padding_bytes,
    layout_kv,
    physical_block_num=None,
):
    if dense_key.dim() != 4 or dense_value.dim() != 4 or dense_k_scale.dim() != 3:
        raise ValueError("dense key/value/k_scale should have ranks 4/4/3")
    if (
        dense_key.shape[:3] != dense_value.shape[:3]
        or dense_key.shape[:3] != dense_k_scale.shape
    ):
        raise ValueError("dense key/value/k_scale B/S/N dimensions should match")

    batch, _, n2, d_size = dense_key.shape
    dv_size = dense_value.shape[-1]
    if physical_block_num is None:
        block_num = int(block_table.max().item()) + 1
    else:
        if not isinstance(physical_block_num, int) or physical_block_num <= 0:
            raise ValueError(
                f"physical_block_num should be a positive int, got {physical_block_num}"
            )
        block_num = physical_block_num
    min_block = int(block_table.min().item())
    max_block = int(block_table.max().item())
    if min_block < 0 or max_block >= block_num:
        raise ValueError(
            f"block_table values should be in [0, {block_num - 1}], "
            f"got min={min_block}, max={max_block}"
        )
    meta = make_combined_kv_meta(
        block_num, block_size, n2, d_size, dv_size, block_padding_bytes, layout_kv
    )
    storage = torch.zeros((block_num * meta["pa_block_stride"],), dtype=torch.uint8)
    key, value, k_scale = make_combined_kv_views(storage, meta)

    for batch_idx in range(batch):
        seq_len = seqused_kv[batch_idx]
        for logical_block in range(block_table.shape[1]):
            physical_block = int(block_table[batch_idx, logical_block].item())
            start = logical_block * block_size
            end = min(start + block_size, seq_len)
            token_count = end - start
            if token_count <= 0:
                continue
            key[physical_block, :, :token_count].copy_(
                dense_key[batch_idx, start:end].permute(1, 0, 2)
            )
            value[physical_block, :, :token_count].copy_(
                dense_value[batch_idx, start:end].permute(1, 0, 2)
            )
            k_scale[physical_block, :, :token_count, 0].copy_(
                dense_k_scale[batch_idx, start:end].permute(1, 0)
            )

    return storage, meta


def assert_combined_kv_views(storage, meta):
    key, value, k_scale = make_combined_kv_views(storage, meta)
    storage_ptr = storage.untyped_storage().data_ptr()
    if not (
        key.untyped_storage().data_ptr()
        == value.untyped_storage().data_ptr()
        == storage_ptr
    ):
        raise AssertionError("key and value should share the combined KV storage")
    if k_scale.untyped_storage().data_ptr() != storage_ptr:
        raise AssertionError("k_scale should share the combined KV storage")

    offsets = (
        key.data_ptr() - storage.data_ptr(),
        value.data_ptr() - storage.data_ptr(),
        k_scale.data_ptr() - storage.data_ptr(),
    )
    expected_offsets = (
        meta["key_offset_bytes"],
        meta["value_offset_bytes"],
        meta["k_scale_offset_bytes"],
    )
    if offsets != expected_offsets:
        raise AssertionError(
            f"unexpected combined KV view offsets: {offsets}, expected {expected_offsets}"
        )

    expected_key_stride = (
        meta["pa_block_stride"],
        meta["block_size"] * meta["d_size"],
        meta["d_size"],
        1,
    )
    expected_value_stride = (
        meta["pa_block_stride"],
        meta["block_size"] * meta["dv_size"],
        meta["dv_size"],
        1,
    )
    expected_k_scale_stride = (
        meta["pa_block_stride"] // FP32_BYTES,
        meta["block_size"],
        1,
        1,
    )
    if key.stride() != expected_key_stride:
        raise AssertionError(
            f"unexpected key stride: {key.stride()}, expected {expected_key_stride}"
        )
    if value.stride() != expected_value_stride:
        raise AssertionError(
            f"unexpected value stride: {value.stride()}, expected {expected_value_stride}"
        )
    if k_scale.stride() != expected_k_scale_stride:
        raise AssertionError(
            f"unexpected k_scale stride: {k_scale.stride()}, expected {expected_k_scale_stride}"
        )
    return key, value, k_scale
