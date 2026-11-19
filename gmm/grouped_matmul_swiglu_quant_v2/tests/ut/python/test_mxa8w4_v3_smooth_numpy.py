#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU only. From repo root:

python -B -m unittest discover -s gmm/grouped_matmul_swiglu_quant_v2/tests/ut/python -p test_mxa8w4_v3_smooth_numpy.py -v

Requires numpy only. Does not import torch, torch_npu, ATK, or an NPU kernel.
"""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


REFERENCE = Path(__file__).resolve().parents[2] / "assets/mxa8w4_v3_smooth_numpy.py"
SPEC = importlib.util.spec_from_file_location("mxa8w4_v3_smooth_numpy", REFERENCE)
ref = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ref
SPEC.loader.exec_module(ref)


def fixture(counts=(0, 17, 0, 127, 1), k=96, n=128, seed=20261024):
    rng = np.random.default_rng(seed)
    e, m = len(counts), sum(counts)
    # Exact finite FP8 byte patterns, including both signs and zero.
    x = rng.choice(np.array([0, 32, 40, 48, 56, 160, 168, 176, 184], np.uint8), (m, k))
    w = rng.choice(np.array([-2, -1, -0.5, 0, 0.5, 1, 2], np.float32), (e, k, n))
    sa = rng.integers(123, 128, (m, (k + 63) // 64, 2), dtype=np.uint8)
    sb = rng.integers(123, 128, (e, (k + 63) // 64, n, 2), dtype=np.uint8)
    return x, w, sa, sb, np.array(counts, np.int64)


def scalar_quant_normal_range(rounded, scale_alg):
    """Independent finite/normal-range byte oracle; not production algorithm."""
    table = ref.decode_e4m3(np.arange(127, dtype=np.uint8))
    m, n = rounded.shape
    out = np.zeros((m, n), dtype=np.uint8)
    scales = np.zeros((m, (n + 63) // 64, 2), dtype=np.uint8)
    for row in range(m):
        for start in range(0, n, 32):
            block = rounded[row, start : start + 32]
            maximum = max(map(lambda value: abs(float(value)), block))
            if maximum == 0:
                if scale_alg == 0:
                    out[row, start : start + 32] = (
                        np.signbit(block).astype(np.uint8) << 7
                    )
                continue
            exponent = (
                math.floor(math.log2(maximum)) - 8
                if scale_alg == 0
                else math.ceil(math.log2(maximum / 448))
            )
            scales.reshape(m, -1)[row, start // 32] = exponent + 127
            for j, value in enumerate(block):
                scaled = float(value) / (2.0**exponent)
                distances = np.abs(table.astype(np.float64) - abs(scaled))
                choices = np.flatnonzero(distances == distances.min())
                even = choices[choices % 2 == 0]
                code = int(even[0] if even.size else choices[0])
                out[row, start + j] = code | (128 if np.signbit(value) else 0)
    return out, scales


class Mxa8w4V3SmoothNumpyTest(unittest.TestCase):
    def test_no_torch_runtime(self):
        self.assertFalse(
            any(name == "torch" or name.startswith("torch.") for name in sys.modules)
        )
        self.assertNotIn("torch_npu", sys.modules)

    def test_fp4_pack_unpack_all_codes_and_0x22(self):
        packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
        decoded = ref.unpack_fp4(packed)
        np.testing.assert_array_equal(ref.pack_fp4(decoded), packed)
        np.testing.assert_array_equal(
            ref.unpack_fp4(np.array([0x22], np.uint8)), [1, 1]
        )
        self.assertTrue(np.signbit(ref.unpack_fp4(np.array([0x08], np.uint8))[0]))
        with self.assertRaises(ValueError):
            ref.pack_fp4(np.array([0.7, 1], np.float32))

    def test_fp8_and_e8m0_known_bytes(self):
        np.testing.assert_array_equal(
            ref.decode_e4m3(np.array([0x38, 0xB8, 0x7E, 1], np.uint8)),
            [1, -1, 448, 2**-9],
        )
        self.assertTrue(np.isnan(ref.decode_e4m3(np.array([127], np.uint8))[0]))
        np.testing.assert_array_equal(
            ref.decode_e8m0(np.array([0, 126, 127, 128], np.uint8)),
            np.array([2**-127, 0.5, 1, 2], np.float32),
        )

    def test_bf16_rne_ties_and_specials(self):
        bits = np.array(
            [
                0x3F808000,
                0x3F818000,
                0xBF808000,
                0xBF818000,
                0x7F800000,
                0xFF800000,
                0x7F800001,
            ],
            np.uint32,
        )
        actual = ref.bf16_rne(bits.view(np.float32))
        np.testing.assert_array_equal(
            actual[:6].view(np.uint32),
            [0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000, 0x7F800000, 0xFF800000],
        )
        self.assertTrue(np.isnan(actual[-1]))

    def test_mode2_clamps_bias_alpha_and_clamped_exponent(self):
        mm = np.array([[20, -20, 2, -2, 20, -20, 2, -2]], np.float32)
        actual = ref.mode2(mm, clamp_limit=3, alpha=0.25, bias=0.75)
        expected = [
            min(a, 3) / (1 + math.exp(-0.25 * min(a, 3))) * (min(max(g, -3), 3) + 0.75)
            for a, g in zip(mm[0, :4], mm[0, 4:])
        ]
        np.testing.assert_allclose(actual[0], expected, rtol=2e-7)
        original_exponent = 3 / (1 + math.exp(-0.25 * 20)) * 3.75
        self.assertGreater(abs(float(actual[0, 0]) - original_exponent), 1)

    def test_left_above_clamp_through_smooth_bf16_and_quant(self):
        x = np.full((1, 64), 0x38, np.uint8)
        w = np.ones((1, 64, 128), np.float32)
        w[:, :, 64:] = 0.5
        sa = np.full((1, 1, 2), 127, np.uint8)
        sb = np.full((1, 1, 128, 2), 127, np.uint8)
        smooth = np.linspace(0.25, 2, 64, dtype=np.float32).reshape(1, 64)
        # GMM left=64 > clamp=3, right=32; sigmoid must use 3 as well.
        expected_act = np.float32(3 / (1 + math.exp(-0.25 * 3)) * 3.75)
        expected_smoothed = expected_act * smooth
        expected_bf16 = ref.bf16_rne(expected_smoothed)
        wrong_act = np.float32(3 / (1 + math.exp(-0.25 * 64)) * 3.75)
        for alg in (0, 1):
            with self.subTest(scale_alg=alg):
                result = ref.golden(
                    x,
                    w,
                    sa,
                    sb,
                    [1],
                    smooth_scale=smooth,
                    clamp_limit=3,
                    alpha=0.25,
                    bias=0.75,
                    scale_alg=alg,
                )
                np.testing.assert_array_equal(result.matmul[:, :64], 64)
                np.testing.assert_array_equal(result.matmul[:, 64:], 32)
                np.testing.assert_allclose(result.swiglu, expected_act, rtol=2e-7)
                np.testing.assert_allclose(
                    result.smoothed, expected_smoothed, rtol=2e-7
                )
                np.testing.assert_array_equal(result.bf16, expected_bf16)
                expected_y, expected_s = scalar_quant_normal_range(expected_bf16, alg)
                np.testing.assert_array_equal(result.output, expected_y)
                np.testing.assert_array_equal(result.output_scale, expected_s)
                wrong_y, wrong_s = scalar_quant_normal_range(
                    ref.bf16_rne(wrong_act * smooth), alg
                )
                self.assertTrue(
                    np.any(result.output != wrong_y)
                    or np.any(result.output_scale != wrong_s)
                )

    def test_random_groups_smoothing_and_quant_bytes(self):
        data = fixture()
        x, w, sa, sb, counts = data
        rng = np.random.default_rng(73)
        random_smooth = rng.uniform(-2, 2, (len(counts), w.shape[-1] // 2)).astype(
            np.float32
        )
        different = np.broadcast_to(
            np.array([0, 0.25, 19, 2, -3], np.float32)[:, None], random_smooth.shape
        ).copy()
        for alg in (0, 1):
            baseline = ref.golden(*data, group_list_type=1, scale_alg=alg)
            for name, smooth in (
                ("none", None),
                ("one", np.ones_like(random_smooth)),
                ("zero", np.zeros_like(random_smooth)),
                ("random", random_smooth),
                ("expert", different),
            ):
                with self.subTest(scale_alg=alg, smooth=name):
                    actual = ref.golden(
                        *data, group_list_type=1, smooth_scale=smooth, scale_alg=alg
                    )
                    cumulative = ref.golden(
                        x,
                        w,
                        sa,
                        sb,
                        np.cumsum(counts),
                        smooth_scale=smooth,
                        scale_alg=alg,
                    )
                    np.testing.assert_array_equal(
                        actual.group_offsets, [0, 0, 17, 17, 144, 145]
                    )
                    np.testing.assert_array_equal(actual.output, cumulative.output)
                    np.testing.assert_array_equal(
                        actual.output_scale, cumulative.output_scale
                    )
                    # Independent group/scale indexing, using exact dyadic inputs.
                    for expert, (start, end) in enumerate(
                        zip(actual.group_offsets[:-1], actual.group_offsets[1:])
                    ):
                        expected_mm = np.zeros((end - start, w.shape[-1]), np.float64)
                        for ki in range(x.shape[1]):
                            a = ref.decode_e4m3(x[start:end, ki]).astype(np.float64)
                            a *= np.exp2(
                                sa[start:end, ki // 64, (ki // 32) % 2].astype(int)
                                - 127
                            )
                            b = w[expert, ki].astype(np.float64)
                            b *= np.exp2(
                                sb[expert, ki // 64, :, (ki // 32) % 2].astype(int)
                                - 127
                            )
                            expected_mm += a[:, None] * b
                        np.testing.assert_array_equal(
                            actual.matmul[start:end], expected_mm.astype(np.float32)
                        )
                        expected = baseline.swiglu[start:end].copy()
                        if smooth is not None:
                            expected *= smooth[expert]
                        np.testing.assert_array_equal(
                            actual.smoothed[start:end], expected
                        )
                    expected_y, expected_s = scalar_quant_normal_range(actual.bf16, alg)
                    np.testing.assert_array_equal(actual.output, expected_y)
                    np.testing.assert_array_equal(actual.output_scale, expected_s)
                    self.assertEqual(actual.output.shape, (145, 64))
                    self.assertEqual(actual.output_scale.shape, (145, 1, 2))
                    self.assertEqual(actual.output.dtype, np.uint8)
                    self.assertEqual(actual.output_scale.dtype, np.uint8)
                    if name in ("none", "one"):
                        np.testing.assert_array_equal(actual.output, baseline.output)
                        np.testing.assert_array_equal(
                            actual.output_scale, baseline.output_scale
                        )
                    if name == "zero":
                        self.assertTrue(np.all((actual.output & 127) == 0))
                        self.assertTrue(np.all(actual.output_scale == 0))

    def test_smooth_before_bf16_changes_final_output(self):
        data = fixture(counts=(3, 0, 2), seed=98)
        smooth = np.random.default_rng(81).uniform(0.2, 2, (3, 64)).astype(np.float32)
        for alg in (0, 1):
            with self.subTest(scale_alg=alg):
                result = ref.golden(
                    *data, group_list_type=1, smooth_scale=smooth, scale_alg=alg
                )
                wrong = ref.bf16_rne(result.swiglu)
                for expert, (start, end) in enumerate(
                    zip(result.group_offsets[:-1], result.group_offsets[1:])
                ):
                    wrong[start:end] *= smooth[expert]
                wrong = ref.bf16_rne(wrong)
                np.testing.assert_array_equal(
                    result.bf16, ref.bf16_rne(result.smoothed)
                )
                self.assertTrue(np.any(result.bf16 != wrong))
                wrong_y, wrong_s = ref.quantize_bf16(wrong, alg)
                self.assertTrue(
                    np.any(result.output != wrong_y)
                    or np.any(result.output_scale != wrong_s)
                )

    def test_scale_alg_thresholds_and_fp8_ties(self):
        x = np.repeat(np.array([256, 448, 450, 512], np.float32)[:, None], 64, axis=1)
        _, ocp = ref.quantize_bf16(x, 0)
        _, cublas = ref.quantize_bf16(x, 1)
        np.testing.assert_array_equal(ocp[:, 0, 0], [127, 127, 127, 128])
        np.testing.assert_array_equal(cublas[:, 0, 0], [127, 127, 128, 128])
        ties = np.tile(
            np.array([1.0625, 1.1875, -1.0625, -1.1875, 448, 500], np.float32), (1, 1)
        )
        encoded = ref._QUANT["_float_to_fp8_bytes"](
            ties, exp_bits=4, mant_bits=3, bias=7, max_exp_field=15, max_mant=6
        )
        np.testing.assert_array_equal(encoded[0], [0x38, 0x3A, 0xB8, 0xBA, 0x7E, 0x7E])

    def test_fp8_subnormal_carry_to_min_normal(self):
        x = np.zeros((1, 64), np.float32)
        x[0, :7] = [
            448,
            7 * 2**-9,
            7.5 * 2**-9,
            2**-6,
            -7 * 2**-9,
            -7.5 * 2**-9,
            -(2**-6),
        ]
        for alg in (0, 1):
            output, _ = ref.quantize_bf16(x, alg)
            np.testing.assert_array_equal(output[0, :7], [0x7E, 7, 8, 8, 135, 136, 136])

    def test_empty_experts_m_tails_and_scale_pair_padding(self):
        for counts in ((0, 0, 0), (1,), (15, 0, 2), (0, 129, 0)):
            for alg in (0, 1):
                with self.subTest(counts=counts, scale_alg=alg):
                    result = ref.golden(
                        *fixture(counts=counts, k=64, n=64),
                        group_list_type=1,
                        scale_alg=alg,
                    )
                    self.assertEqual(result.output.shape, (sum(counts), 32))
                    self.assertEqual(result.output_scale.shape, (sum(counts), 1, 2))
                    self.assertTrue(np.all(result.output_scale[:, :, 1] == 0))

    def test_reject_bad_groups_smooth_and_scales(self):
        x, w, sa, sb, counts = fixture(counts=(2, 0, 1))
        for groups in ([2, 1, 3], [0, 0, 2], [-1, 2, 3], [3]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                ref.golden(x, w, sa, sb, groups)
        for smooth in (
            np.ones(64, np.float32),
            np.ones((3, 64), np.float64),
            np.full((3, 64), np.nan, np.float32),
        ):
            with self.assertRaises(ValueError):
                ref.golden(x, w, sa, sb, counts, group_list_type=1, smooth_scale=smooth)
        with self.assertRaises(ValueError):
            ref.golden(x, w, sa[:, :, :1], sb, counts, group_list_type=1)
        with self.assertRaises(ValueError):
            ref.golden(x, w, sa, sb, counts, group_list_type=1, scale_alg=2)


if __name__ == "__main__":
    unittest.main()
