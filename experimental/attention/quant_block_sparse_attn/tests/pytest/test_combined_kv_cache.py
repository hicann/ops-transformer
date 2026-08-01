#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import pytest

import combined_kv_cache


def _sample_dense_inputs():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required for combined KV cache tests")
    fp8_dtype = torch.float8_e4m3fn
    dense_key = torch.arange(2 * 5 * 2 * 8, dtype=torch.float32).reshape(2, 5, 2, 8).to(fp8_dtype)
    dense_value = (-torch.arange(2 * 5 * 2 * 8, dtype=torch.float32)).reshape(2, 5, 2, 8).to(fp8_dtype)
    dense_k_scale = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    block_table = torch.tensor([[1, 0], [3, 2]], dtype=torch.int32)
    return dense_key, dense_value, dense_k_scale, block_table


def test_combined_kv_cache_views_and_segments():
    dense_key, dense_value, dense_k_scale, block_table = _sample_dense_inputs()
    storage, meta = combined_kv_cache.pack_combined_kv_cache(
        dense_key, dense_value, dense_k_scale, block_table, block_size=4,
        block_padding_bytes=0, layout_kv="PA_BNSD")
    key, value, k_scale = combined_kv_cache.assert_combined_kv_views(storage, meta)

    assert meta["packing"] == combined_kv_cache.PACKING
    assert meta["key_segment_bytes"] == 2 * 4 * 8
    assert meta["value_segment_bytes"] == 2 * 4 * 8
    assert meta["k_scale_segment_bytes"] == 2 * 4 * 4
    assert meta["pa_block_stride"] == 2 * 4 * 8 + 2 * 4 * 8 + 2 * 4 * 4
    assert key.stride() == (meta["pa_block_stride"], 4 * 8, 8, 1)
    assert value.stride() == (meta["pa_block_stride"], 4 * 8, 8, 1)
    assert k_scale.shape == (4, 2, 4, 1)
    assert k_scale.stride() == (meta["pa_block_stride"] // 4, 4, 1, 1)

    for batch_idx in range(2):
        for logical_block in range(2):
            physical_block = int(block_table[batch_idx, logical_block])
            start = logical_block * 4
            end = min(start + 4, 5)
            count = end - start
            torch.testing.assert_close(
                key[physical_block, :, :count].to(torch.float32),
                dense_key[batch_idx, start:end].permute(1, 0, 2).to(torch.float32))
            torch.testing.assert_close(
                value[physical_block, :, :count].to(torch.float32),
                dense_value[batch_idx, start:end].permute(1, 0, 2).to(torch.float32))
            torch.testing.assert_close(
                k_scale[physical_block, :, :count, 0],
                dense_k_scale[batch_idx, start:end].permute(1, 0))

            block_base = physical_block * meta["pa_block_stride"]
            key_expected = torch.zeros((2, 4, 8), dtype=key.dtype)
            value_expected = torch.zeros((2, 4, 8), dtype=value.dtype)
            k_scale_expected = torch.zeros((2, 4), dtype=k_scale.dtype)
            key_expected[:, :count].copy_(dense_key[batch_idx, start:end].permute(1, 0, 2))
            value_expected[:, :count].copy_(dense_value[batch_idx, start:end].permute(1, 0, 2))
            k_scale_expected[:, :count].copy_(dense_k_scale[batch_idx, start:end].permute(1, 0))

            key_slice = storage[block_base:block_base + meta["key_segment_bytes"]]
            value_base = block_base + meta["value_offset_bytes"]
            value_slice = storage[value_base:value_base + meta["value_segment_bytes"]]
            k_scale_base = block_base + meta["k_scale_offset_bytes"]
            k_scale_slice = storage[k_scale_base:k_scale_base + meta["k_scale_segment_bytes"]]

            assert torch.equal(key_slice, key_expected.contiguous().view(torch.uint8).flatten())
            assert torch.equal(value_slice, value_expected.contiguous().view(torch.uint8).flatten())
            torch.testing.assert_close(
                k_scale_slice.view(torch.float32).reshape(2, 4),
                k_scale_expected)


def test_combined_kv_cache_storage_serialization(tmp_path):
    dense_key, dense_value, dense_k_scale, block_table = _sample_dense_inputs()
    storage, meta = combined_kv_cache.pack_combined_kv_cache(
        dense_key, dense_value, dense_k_scale, block_table, block_size=4,
        block_padding_bytes=0, layout_kv="PA_BNSD")

    path = tmp_path / "combined_kv.pt"
    torch.save({"storage": storage, "meta": meta}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    key, value, k_scale = combined_kv_cache.assert_combined_kv_views(loaded["storage"], loaded["meta"])

    assert key.shape == (4, 2, 4, 8)
    assert value.shape == key.shape
    assert k_scale.shape == (4, 2, 4, 1)
    assert loaded["meta"]["packing"] == combined_kv_cache.PACKING
    assert loaded["meta"]["pa_block_stride"] == meta["pa_block_stride"]
