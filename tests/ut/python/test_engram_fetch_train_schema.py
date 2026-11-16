#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the public engram_fetch_train schema on Meta and NPU dispatch."""

import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cann_ops_transformer.ops.mc2.common.elastic_buffer import (
    _ENGRAM_SF_DTYPES,
)


OP = torch.ops.cann_ops_transformer.engram_fetch_train


def call_args(sf_dtype, sf_packs, device, num_tokens=128):
    return (
        torch.empty(4100, dtype=torch.int32, device=device),
        torch.empty(num_tokens, dtype=torch.int32, device=device),
        128,
        16384,
        15,
        torch.empty(1, dtype=torch.int64, device=device),
        sf_dtype,
        sf_packs,
        torch.empty(1, dtype=torch.int64, device=device),
        128,
        419430400,
        4,
    )


class EngramFetchTrainSchemaTest(unittest.TestCase):
    def assert_invalid(self, sf_dtype, sf_packs, message):
        with FakeTensorMode():
            args = call_args(sf_dtype, sf_packs, device="cpu")
            with self.assertRaises(RuntimeError) as context:
                OP(*args)
        self.assertIn(message, str(context.exception))

        if not torch.npu.is_available():
            self.skipTest("The real dispatch path requires an NPU device")
        torch.npu.set_device(0)
        args = call_args(sf_dtype, sf_packs, device="npu")
        with self.assertRaises(RuntimeError) as context:
            OP(*args)
        self.assertIn(message, str(context.exception))

    def test_meta_and_real_reject_negative_sf_packs(self):
        self.assert_invalid(
            torch.float32,
            -1,
            "sf_packs must be non-negative, got -1",
        )

    def test_meta_and_real_reject_invalid_sf_dtype(self):
        for sf_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(sf_dtype=sf_dtype):
                self.assert_invalid(
                    sf_dtype,
                    0,
                    "sf_dtype must be float32 or float8_e8m0fnu",
                )

    def test_meta_and_real_accept_supported_sf_dtypes(self):
        for sf_dtype in _ENGRAM_SF_DTYPES:
            with self.subTest(sf_dtype=sf_dtype), FakeTensorMode():
                result = OP(*call_args(sf_dtype, 40, device="cpu", num_tokens=0))
                self.assertEqual(result[1].shape, (0, 40))
                self.assertEqual(result[1].dtype, sf_dtype)

            if not torch.npu.is_available():
                self.skipTest("The real dispatch path requires an NPU device")
            torch.npu.set_device(0)
            result = OP(*call_args(sf_dtype, 40, device="npu", num_tokens=0))
            self.assertEqual(result[1].shape, (0, 40))
            self.assertEqual(result[1].dtype, sf_dtype)
            self.assertEqual(result[1].device.type, "npu")

    def test_meta_and_real_accept_zero_sf_packs(self):
        with FakeTensorMode():
            result = OP(*call_args(torch.float32, 0, device="cpu", num_tokens=0))
            self.assertEqual(result[1].shape, (0,))
            self.assertEqual(result[1].dtype, torch.float32)

        if not torch.npu.is_available():
            self.skipTest("The real dispatch path requires an NPU device")
        torch.npu.set_device(0)
        result = OP(*call_args(torch.float32, 0, device="npu", num_tokens=0))
        self.assertEqual(result[1].shape, (0,))
        self.assertEqual(result[1].dtype, torch.float32)
        self.assertEqual(result[1].device.type, "npu")


if __name__ == "__main__":
    unittest.main()
