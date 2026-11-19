# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Ascend 950/CANN smoke test for the actual TorchExtension and descriptor path."""

import itertools
import unittest

import torch


class TestA8W4BridgeNpu(unittest.TestCase):
    def test_transposed_weight_identity_smooth_and_strided_scale(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest("Requires torch_npu and Ascend 950")
        if not torch.npu.is_available():
            self.skipTest("Requires an available Ascend 950")
        if "950" not in torch.npu.get_device_name():
            self.skipTest("MX A8W4 V3 requires Ascend 950")
        import cann_ops_transformer

        experts, m, k = 4, 17, 128
        # Repeated nonzero FP4 nibbles keep the identity-smooth comparison exact.
        fp4_ones_byte = 0x22
        group_list = torch.tensor([0, 8, 0, 9], dtype=torch.int64).npu()
        x = (torch.arange(m * k).reshape(m, k) % 5 - 2).to(torch.float8_e4m3fn).npu()
        x_scale = torch.full((m, k // 64, 2), 127, dtype=torch.uint8).npu()
        strided_x_scale = x_scale.repeat_interleave(2, dim=-1)[..., ::2]
        self.assertFalse(strided_x_scale.is_contiguous())

        def raw(tensor):
            return tensor.view(torch.uint8).cpu()

        for n in (128, 384):
            expected = None
            for transpose in (True,):
                shape = (experts, n, k // 2) if transpose else (experts, k, n // 2)
                # Convert packed bytes to NZ before attaching the FP4 dtype.
                weight = torch.full(shape, fp4_ones_byte, dtype=torch.int8).npu()
                weight = torch_npu.npu_format_cast(weight, 29)
                weight = weight.view(torch.float4_e2m1fn_x2)
                scale_shape = (
                    (experts, n, k // 64, 2) if transpose else (experts, k // 64, n, 2)
                )
                scale = torch.full(scale_shape, 127, dtype=torch.uint8).npu()
                if transpose:
                    weight = weight.transpose(-1, -2)
                    scale = scale.transpose(-3, -2)
                for smooth, scale_a in itertools.product(
                    (None, torch.ones((experts, n // 2), dtype=torch.float32).npu()),
                    (x_scale, strided_x_scale),
                ):
                    with self.subTest(
                        n=n,
                        transpose=transpose,
                        smooth=smooth is not None,
                        contiguous_scale=scale_a.is_contiguous(),
                    ):
                        y, ys = cann_ops_transformer.grouped_matmul_swiglu_quant(
                            x,
                            [weight],
                            [scale],
                            scale_a,
                            group_list,
                            swiglu_mode=2,
                            group_list_type=1,
                            smooth_scale=smooth,
                        )
                        torch.npu.synchronize()
                        self.assertEqual(tuple(y.shape), (m, n // 2))
                        self.assertEqual(tuple(ys.shape), (m, (n + 127) // 128, 2))
                        actual = (raw(y), raw(ys))
                        if expected is None:
                            expected = actual
                        else:
                            self.assertTrue(torch.equal(actual[0], expected[0]))
                            self.assertTrue(torch.equal(actual[1], expected[1]))


if __name__ == "__main__":
    unittest.main()
