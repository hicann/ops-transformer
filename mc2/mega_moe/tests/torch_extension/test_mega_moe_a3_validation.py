# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest


class TestMegaMoeA3Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch_npu
        except ModuleNotFoundError as exc:
            if exc.name not in ("torch", "torch_npu"):
                raise
            raise unittest.SkipTest("Requires torch and torch_npu") from exc
        if (
            not torch.npu.is_available()
            or "Ascend910_93" not in torch.npu.get_device_name()
        ):
            raise unittest.SkipTest("Requires an Ascend A3 device")

        from cann_ops_transformer.ops.mc2.mega_moe.mega_moe import (
            _BF16_DTYPE_VALUE,
            _get_mega_moe_ccl_buffer_size,
        )

        cls.torch = torch
        cls.torch_npu = torch_npu
        cls.bf16_dtype_value = _BF16_DTYPE_VALUE
        cls.mega_moe = torch.ops.cann_ops_transformer.npu_mega_moe
        cls.query_buffer = staticmethod(_get_mega_moe_ccl_buffer_size)

    def test_shared_weight_inputs_are_checked_before_acl_conversion(self):
        torch = self.torch
        w1 = torch.empty((1024, 1024), dtype=torch.bfloat16, device="npu")
        w2 = torch.empty((512, 1024), dtype=torch.bfloat16, device="npu")
        half_w1 = torch.empty_like(w1, dtype=torch.float16)
        half_w2 = torch.empty_like(w2, dtype=torch.float16)
        x = torch.empty((1, 1024), dtype=torch.bfloat16, device="npu")
        ids = torch.zeros((1, 1), dtype=torch.int32, device="npu")
        scores = torch.ones((1, 1), dtype=torch.float32, device="npu")
        # An invalid context rank prevents the old implementation from launching the kernel.
        context = torch.zeros((1, 1), dtype=torch.int32, device="npu")
        cases = (
            ([w1] * 3, [w2] * 2, None, r"at most 2 local shared experts"),
            ([w1] * 2, [w2] * 3, None, r"at most 2 local shared experts"),
            ([w1, half_w1], [w2, w2], None, r"shared_l1_weights\[1\] dtype"),
            ([w1, w1], [w2, half_w2], None, r"shared_l2_weights\[1\] dtype"),
            (
                [w1, w1],
                [half_w2, half_w2],
                self.bf16_dtype_value,
                r"shared_l2_weights\[0\] dtype",
            ),
        )
        for shared_w1, shared_w2, w2_type, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                self.mega_moe(
                    context,
                    x,
                    ids,
                    scores,
                    [w1, w1],
                    [w2, w2],
                    4,
                    2,
                    32 * 1024 * 1024,
                    shared_weight1=shared_w1,
                    shared_weight2=shared_w2,
                    shared_weight2_type=w2_type,
                )

    def test_packed_int4_shared_weight_requires_int32_storage_for_every_entry(self):
        torch = self.torch
        config = torch.npu.config
        allow_internal = (
            self.torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
        )
        self.addCleanup(setattr, config, "allow_internal_format", allow_internal)
        config.allow_internal_format = True
        packed_w1 = self.torch_npu.npu_format_cast(
            torch.zeros((1024, 512), dtype=torch.int8, device="npu"), 29
        ).view(torch.int32)
        packed_w2 = self.torch_npu.npu_format_cast(
            torch.zeros((512, 512), dtype=torch.int8, device="npu"), 29
        ).view(torch.int32)
        int8_w2 = self.torch_npu.npu_format_cast(
            torch.zeros((512, 1024), dtype=torch.int8, device="npu"), 29
        )
        for weight in (packed_w1, packed_w2, int8_w2):
            self.assertEqual(self.torch_npu.get_npu_format(weight), 29)
        context = torch.zeros((1, 1), dtype=torch.int32, device="npu")
        x = torch.empty((1, 1024), dtype=torch.bfloat16, device="npu")
        ids = torch.zeros((1, 1), dtype=torch.int32, device="npu")
        scores = torch.ones((1, 1), dtype=torch.float32, device="npu")
        with self.assertRaisesRegex(RuntimeError, r"shared_l2_weights\[1\] dtype"):
            self.mega_moe(
                context,
                x,
                ids,
                scores,
                [packed_w1] * 2,
                [packed_w2] * 2,
                4,
                2,
                32 * 1024 * 1024,
                shared_weight1=[packed_w1] * 2,
                shared_weight2=[packed_w2, int8_w2],
                dispatch_quant_mode=2,
            )

    def test_buffer_query_matches_tiling_hidden_and_topk_bounds(self):
        for hidden, topk in ((1024, 1), (8192, 16)):
            with self.subTest(hidden=hidden, topk=topk):
                self.assertGreater(self.query_buffer(2, 32, 1, topk, hidden, 512), 0)

        cases = (
            (512, 1, 32, "hidden"),
            (1152, 1, 32, "hidden"),
            (8704, 1, 32, "hidden"),
            (1024, 0, 32, "num_topk"),
            (1024, 17, 32, "num_topk"),
            (1024, 9, 8, "num_topk"),
        )
        for hidden, topk, experts, error in cases:
            with (
                self.subTest(hidden=hidden, topk=topk, experts=experts),
                self.assertRaisesRegex(RuntimeError, error),
            ):
                self.query_buffer(2, experts, 1, topk, hidden, 512)


if __name__ == "__main__":
    unittest.main()
