#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU reference regressions; device coverage is in ttk_aclnn_normal_quant.csv."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "indexer_golden", Path(__file__).resolve().parents[2] / "assets" / "golden.py"
)
golden = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(golden)
golden._ensure_dtypes()


class QuantGoldenTest(unittest.TestCase):
    def normal(self, x, dtype, round_scale=False):
        cache = np.zeros((1, 1, 1, 32), dtype=dtype)
        scale = np.ones((1, 1, 1, 1), dtype=np.float32)
        return golden.IndexerQuantCacheTestSpec.golden(
            cache,
            scale,
            np.resize(x, (1, 32)),
            np.array([0], np.int32),
            quant_mode=1,
            round_scale=round_scale,
        )

    def test_mxfp8_nan_scale(self):
        q, scale = golden._encode_mxfp8_row(
            np.full(32, np.nan, np.float32), 32, golden.F8E4M3, np.float32(448), True
        )
        self.assertEqual(int(scale[0]), 255)
        self.assertTrue(np.isnan(q.astype(np.float32)).all())

    def test_normal_inf(self):
        q, scale = self.normal(np.array([np.inf, -np.inf], np.float32), golden.F8E5M2)
        np.testing.assert_array_equal(
            q.astype(np.float32).ravel()[:2], [np.inf, -np.inf]
        )
        self.assertEqual(float(scale.item()), 0)

    def test_normal_mixed_nonfinite(self):
        q, scale = self.normal(
            np.array([1, -1, np.inf, np.nan], np.float32), golden.F8E5M2
        )
        self.assertAlmostEqual(float(scale.item()), 1 / 57344)
        np.testing.assert_array_equal(
            q.astype(np.float32).ravel()[:3], [57344, -57344, np.inf]
        )
        self.assertTrue(np.isnan(q.astype(np.float32).ravel()[3]))

    def test_normal_uint8_hifloat_bytes(self):
        for round_scale in (False, True):
            q, scale = self.normal(np.array([1, -1], np.float32), np.uint8, round_scale)
            expected = (
                np.array([32768, -32768], np.float32).astype(golden.HIF8).view(np.uint8)
            )
            np.testing.assert_array_equal(q.ravel()[:2], expected)
            self.assertEqual(float(scale.item()), 1 / 32768)

    def test_normal_tiny_bfloat16(self):
        q, scale = self.normal(
            np.array([np.finfo(np.float32).tiny], dtype=golden.BF16), golden.F8E4M3
        )
        self.assertTrue((q.astype(np.float32) == 448).all())
        self.assertGreater(float(scale.item()), 0)


if __name__ == "__main__":
    unittest.main()
