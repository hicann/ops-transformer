#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math

import numpy as np


HEAD_DIM = 512
PACKED_BYTES = HEAD_DIM // 2
SLOT_BYTES = PACKED_BYTES + 2
SLOT_ROW_BYTES = 288
BF16_DEQUANT_BATCH_ROWS = 16
CENTROIDS = np.array(
    [
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
    ],
    dtype=np.float32,
)


def _float32_to_bf16_bits(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return (
        (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
        >> np.uint32(16)
    ).astype(np.uint16)


def _bf16_bits_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _make_tq4_byte_lut():
    physical_centroids = CENTROIDS[np.arange(16) ^ 8]
    centroid_bits = _float32_to_bf16_bits(physical_centroids).astype(np.uint32)
    packed_bytes = np.arange(256, dtype=np.uint32)
    low_bits = centroid_bits[packed_bytes & 0xF]
    high_bits = centroid_bits[packed_bytes >> 4]
    return (high_bits << np.uint32(16)) | low_bits


def _dequant_byte_lut(slots, scales):
    byte_lut = _make_tq4_byte_lut()
    packed_pairs = byte_lut[slots[:, :PACKED_BYTES]].view(np.uint16)
    centroids = _bf16_bits_to_float32(packed_pairs.reshape(slots.shape[0], HEAD_DIM))
    scaled = (
        centroids * np.asarray(scales, dtype=np.float16).astype(np.float32)[:, None]
    )
    return _float32_to_bf16_bits(scaled)


def _hadamard(order):
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(order)


def test_compact_slot_layout():
    assert PACKED_BYTES == 256
    assert SLOT_BYTES == 258
    assert (1024 - SLOT_BYTES) / 1024 == 0.748046875


def test_signed_nibble_centroid_order():
    physical_codes = np.arange(16, dtype=np.int16)
    signed_codes = np.where(physical_codes < 8, physical_codes, physical_codes - 16)
    gather_indices = signed_codes + 8
    np.testing.assert_array_equal(gather_indices, np.r_[8:16, 0:8])
    assert np.all(np.diff(CENTROIDS[gather_indices]) > 0)


def test_byte_lut_exhaustively_matches_two_bf16_centroid_lookups():
    byte_lut = _make_tq4_byte_lut()
    packed_bytes = np.arange(256, dtype=np.uint32)
    physical_centroid_bits = _float32_to_bf16_bits(CENTROIDS[np.arange(16) ^ 8])

    np.testing.assert_array_equal(
        (byte_lut & np.uint32(0xFFFF)).astype(np.uint16),
        physical_centroid_bits[packed_bytes & 0xF],
    )
    np.testing.assert_array_equal(
        (byte_lut >> np.uint32(16)).astype(np.uint16),
        physical_centroid_bits[packed_bytes >> 4],
    )


def test_byte_lut_dequant_handles_every_batch_tail():
    rng = np.random.default_rng(6)
    for row_count in range(1, BF16_DEQUANT_BATCH_ROWS + 1):
        slots = np.zeros((row_count, SLOT_ROW_BYTES), dtype=np.uint8)
        slots[:, :PACKED_BYTES] = rng.integers(
            0, 256, size=(row_count, PACKED_BYTES), dtype=np.uint8
        )
        scales = rng.uniform(0.125, 8.0, row_count).astype(np.float16)
        actual = _dequant_byte_lut(slots, scales)

        low_codes = slots[:, :PACKED_BYTES] & np.uint8(0xF)
        high_codes = slots[:, :PACKED_BYTES] >> np.uint8(4)
        codes = np.empty((row_count, HEAD_DIM), dtype=np.uint8)
        codes[:, 0::2] = low_codes
        codes[:, 1::2] = high_codes
        centroid_values = _bf16_bits_to_float32(
            _float32_to_bf16_bits(CENTROIDS[codes ^ np.uint8(8)])
        )
        expected = _float32_to_bf16_bits(
            centroid_values * scales.astype(np.float32)[:, None]
        )
        np.testing.assert_array_equal(actual, expected)


def test_byte_lut_scale_rounding_is_within_one_bf16_ulp_of_float_lut():
    finite_fp16 = np.arange(1 << 16, dtype=np.uint16).view(np.float16)
    scales = finite_fp16[np.isfinite(finite_fp16)].astype(np.float32)
    bf16_centroids = _bf16_bits_to_float32(_float32_to_bf16_bits(CENTROIDS))

    for centroid, bf16_centroid in zip(CENTROIDS, bf16_centroids):
        old_bits = _float32_to_bf16_bits(centroid * scales).astype(np.int32)
        optimized_bits = _float32_to_bf16_bits(bf16_centroid * scales).astype(np.int32)
        assert np.max(np.abs(old_bits - optimized_bits)) <= 1


def test_corrected_scale_preserves_quantized_row_norm():
    rng = np.random.default_rng(0)
    latent = rng.standard_normal(HEAD_DIM).astype(np.float32)
    unit = latent / np.linalg.norm(latent)
    ascending = np.sort(CENTROIDS)
    boundaries = (ascending[:-1] + ascending[1:]) * np.float32(0.5)
    ascending_codes = np.sum(unit[:, None] >= boundaries[None, :], axis=1)
    selected = ascending[ascending_codes]
    corrected_scale = np.linalg.norm(latent) / np.linalg.norm(selected)
    reconstructed = selected * corrected_scale
    np.testing.assert_allclose(
        np.linalg.norm(reconstructed), np.linalg.norm(latent), rtol=2e-7
    )


def test_shared_kv_attention_is_hadamard_basis_invariant():
    rng = np.random.default_rng(1)
    transform = _hadamard(HEAD_DIM)
    signs = np.where(np.arange(HEAD_DIM) % 3 == 0, -1.0, 1.0)
    transform = signs[:, None] * transform
    query = rng.standard_normal((3, HEAD_DIM))
    shared_kv = rng.standard_normal((7, HEAD_DIM))
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def attention(q, kv):
        scores = q @ kv.T * scale
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return probabilities @ kv

    reference = attention(query, shared_kv)
    transformed = attention(query @ transform, shared_kv @ transform) @ transform.T
    np.testing.assert_allclose(transformed, reference, rtol=1e-10, atol=1e-10)


def _attention_with_sinks(query, key, value, sinks):
    scores = query @ key.T / math.sqrt(query.shape[-1])
    max_values = np.maximum(scores.max(axis=-1), sinks)
    score_exp = np.exp(scores - max_values[:, None])
    sink_exp = np.exp(sinks - max_values)
    denominator = score_exp.sum(axis=-1) + sink_exp
    output = score_exp @ value / denominator[:, None]
    lse = np.log(denominator) + max_values
    return output, lse


def _folded_shared_kv_attention(query, centroids, scales, sinks):
    scores = (query @ centroids.T) * scales[None, :] / math.sqrt(query.shape[-1])
    max_values = np.maximum(scores.max(axis=-1), sinks)
    score_exp = np.exp(scores - max_values[:, None])
    sink_exp = np.exp(sinks - max_values)
    denominator = score_exp.sum(axis=-1) + sink_exp
    probabilities = score_exp / denominator[:, None]
    output = (probabilities * scales[None, :]) @ centroids
    lse = np.log(denominator) + max_values
    return output, lse


def test_attention_fold_is_equivalent_with_reordered_and_duplicate_rows():
    rng = np.random.default_rng(2)
    query = rng.standard_normal((11, 32))
    codebook_rows = rng.standard_normal((17, 32))
    selected = np.array([8, 3, 8, 16, 1, 12, 0, 12, 5, 4, 9, 2, 15])
    centroids = codebook_rows[selected]
    scales = np.exp(rng.uniform(np.log(2**-8), np.log(2**8), selected.size))
    sinks = rng.standard_normal(query.shape[0])

    reference = _attention_with_sinks(
        query, centroids * scales[:, None], centroids * scales[:, None], sinks
    )
    folded = _folded_shared_kv_attention(query, centroids, scales, sinks)

    np.testing.assert_allclose(folded[0], reference[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(folded[1], reference[1], rtol=2e-12, atol=2e-12)


def test_attention_fold_preserves_original_and_compressed_regions():
    rng = np.random.default_rng(3)
    query = rng.standard_normal((7, 24))
    original = rng.standard_normal((9, 24))
    compressed = rng.standard_normal((15, 24))
    scales = np.exp(rng.uniform(np.log(2**-6), np.log(2**6), compressed.shape[0]))
    sinks = rng.standard_normal(query.shape[0])

    scaled_compressed = compressed * scales[:, None]
    reference = _attention_with_sinks(
        query,
        np.concatenate((original, scaled_compressed)),
        np.concatenate((original, scaled_compressed)),
        sinks,
    )

    original_scores = query @ original.T / math.sqrt(query.shape[-1])
    compressed_scores = (
        (query @ compressed.T) * scales[None, :] / math.sqrt(query.shape[-1])
    )
    scores = np.concatenate((original_scores, compressed_scores), axis=-1)
    max_values = np.maximum(scores.max(axis=-1), sinks)
    score_exp = np.exp(scores - max_values[:, None])
    denominator = score_exp.sum(axis=-1) + np.exp(sinks - max_values)
    probabilities = score_exp / denominator[:, None]
    folded_output = (
        probabilities[:, : original.shape[0]] @ original
        + (probabilities[:, original.shape[0] :] * scales[None, :]) @ compressed
    )
    folded_lse = np.log(denominator) + max_values

    np.testing.assert_allclose(folded_output, reference[0], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(folded_lse, reference[1], rtol=2e-12, atol=2e-12)


def test_padding_scales_are_neutral_before_masking():
    rng = np.random.default_rng(4)
    query = rng.standard_normal((5, 16))
    valid_columns = 19
    padded_columns = 32
    centroids = rng.standard_normal((padded_columns, 16))
    scales = np.ones(padded_columns)
    scales[:valid_columns] = np.exp(rng.uniform(-3.0, 3.0, valid_columns))

    folded_scores = (query @ centroids.T) * scales[None, :]
    folded_scores[:, valid_columns:] = -np.inf
    reference_scores = (
        query @ (centroids[:valid_columns] * scales[:valid_columns, None]).T
    )
    np.testing.assert_allclose(
        folded_scores[:, :valid_columns], reference_scores, rtol=2e-12, atol=2e-12
    )
    assert np.all(np.isneginf(folded_scores[:, valid_columns:]))


def test_fp16_fold_rounding_remains_numerically_close():
    rng = np.random.default_rng(5)
    query = rng.standard_normal((8, 64)).astype(np.float32)
    centroids = rng.standard_normal((41, 64)).astype(np.float32)
    scales = np.exp(
        rng.uniform(np.log(2**-4), np.log(2**4), centroids.shape[0])
    ).astype(np.float16)
    sinks = rng.standard_normal(query.shape[0]).astype(np.float32)

    baseline_kv = (
        (centroids * scales.astype(np.float32)[:, None])
        .astype(np.float16)
        .astype(np.float32)
    )
    folded_kv = centroids.astype(np.float16).astype(np.float32)
    reference = _attention_with_sinks(query, baseline_kv, baseline_kv, sinks)
    folded = _folded_shared_kv_attention(
        query, folded_kv, scales.astype(np.float32), sinks
    )

    output_error = folded[0] - reference[0]
    assert np.linalg.norm(output_error) / np.linalg.norm(reference[0]) < 2e-3
    assert np.max(np.abs(output_error)) / np.max(np.abs(reference[0])) < 2e-3
    np.testing.assert_allclose(folded[1], reference[1], rtol=2e-3, atol=2e-3)


def test_mtp_tile_mask_matches_per_token_sliding_window():
    query_length = 7
    kv_length = 35
    window_left = 11
    tile_query_start = 1
    tile_query_end = 5

    tile_left = max(kv_length - query_length + tile_query_start - window_left, 0)
    tile_right = kv_length - query_length + tile_query_end
    tile_keys = np.arange(tile_left, tile_right + 1)

    tiled_mask = np.empty(
        (tile_query_end - tile_query_start + 1, tile_keys.size), dtype=bool
    )
    for row, query_pos in enumerate(range(tile_query_start, tile_query_end + 1)):
        row_left = max(kv_length - query_length + query_pos - window_left, 0)
        row_right = kv_length - query_length + query_pos
        tiled_mask[row] = (tile_keys < row_left) | (tile_keys > row_right)

    reference = np.stack(
        [
            (tile_keys < max(kv_length - query_length + query_pos - window_left, 0))
            | (tile_keys > kv_length - query_length + query_pos)
            for query_pos in range(tile_query_start, tile_query_end + 1)
        ]
    )
    np.testing.assert_array_equal(tiled_mask, reference)
    assert np.any(tiled_mask[0] != tiled_mask[-1])
