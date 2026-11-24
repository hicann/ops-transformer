# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
import importlib
import os
import unittest
import torch

ops = importlib.import_module(
    os.environ.get("CANN_TRANSFORMER_PACKAGE", "cann_ops_transformer")
)


class DualLevelMeta(unittest.TestCase):
    def inputs(self, transpose=False):
        n = 17 if transpose else 32
        weight = torch.empty(
            (3, n, 257) if transpose else (3, 514, 16), dtype=torch.int8, device="meta"
        )
        weight_level1_scale = torch.empty(
            (3, n, 9, 2) if transpose else (3, 9, n, 2),
            dtype=torch.uint8,
            device="meta",
        )
        if transpose:
            weight = weight.transpose(1, 2)
            weight_level1_scale = weight_level1_scale.transpose(1, 2)
        return dict(
            x=[torch.empty((33, 257), dtype=torch.int8, device="meta")],
            weight=[weight],
            x_level0_scale=[torch.empty((33, 2), device="meta")],
            x_level1_scale=[torch.empty((33, 9, 2), dtype=torch.uint8, device="meta")],
            weight_level0_scale=[torch.empty((3, 2, n), device="meta")],
            weight_level1_scale=[weight_level1_scale],
            group_list=torch.empty(3, dtype=torch.int64, device="meta"),
        )

    def test_output(self):
        for transpose in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(transpose=transpose, dtype=dtype):
                    out = ops.dual_level_grouped_matmul(
                        **self.inputs(transpose), output_dtype=dtype
                    )
                    self.assertEqual(len(out), 1)
                    self.assertEqual(out[0].shape, (33, 17 if transpose else 32))
                    self.assertEqual(out[0].dtype, dtype)

    def test_sparse_groups(self):
        for transpose in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                args = self.inputs(transpose)
                args["group_list"] = torch.empty(
                    (3, 2), dtype=torch.int64, device="meta"
                )
                out = ops.dual_level_grouped_matmul(
                    **args, group_list_type=2, output_dtype=dtype
                )
                self.assertEqual(out[0].shape, (33, 17 if transpose else 32))
                self.assertEqual(out[0].dtype, dtype)

    def test_only_checks_output_inference_inputs(self):
        args = self.inputs()
        args["x_level0_scale"] = [torch.empty((33, 3), device="meta")]
        out = ops.dual_level_grouped_matmul(**args)
        self.assertEqual(out[0].shape, (33, 32))

    def test_missing_output_shape_input(self):
        args = self.inputs()
        args["weight_level1_scale"] = []
        with self.assertRaisesRegex(ValueError, "must each contain one tensor"):
            ops.dual_level_grouped_matmul(**args)


try:
    import torch_npu
except ImportError:
    torch_npu = None


@unittest.skipUnless(torch_npu is not None, "requires torch and torch_npu")
class DualLevelStorageFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("requires an Ascend 950 NPU")
        cls.ops = importlib.import_module(
            os.environ.get("CANN_TRANSFORMER_PACKAGE", "cann_ops_transformer")
        )

    def inputs(self):
        def full(shape, value, dtype):
            tensor = torch.full(shape, value, dtype=dtype, device="npu")
            return torch_npu.npu_format_cast(tensor, 2)

        return dict(
            x=[full((33, 257), 0x22, torch.int8)],
            weight=[
                torch_npu.npu_format_cast(full((3, 514, 16), 0x22, torch.int8), 29)
            ],
            x_level0_scale=[full((33, 2), 0.5, torch.float32)],
            x_level1_scale=[full((33, 9, 2), 127, torch.int8)],
            weight_level0_scale=[full((3, 2, 32), 2.0, torch.float32)],
            weight_level1_scale=[full((3, 9, 32, 2), 127, torch.int8)],
            bias=[full((3, 32), 1.0, torch.float32)],
            group_list=torch.tensor([0, 1, 33], dtype=torch.int64, device="npu"),
            output_dtype=torch.float16,
        )

    def test_nd_inputs_compute_expected_output(self):
        output = self.ops.dual_level_grouped_matmul(**self.inputs())[0]
        torch.testing.assert_close(
            output.cpu(), torch.full((33, 32), 515, dtype=torch.float16), rtol=0, atol=0
        )

    def test_bad_scale_shape_rejected_by_acl(self):
        args = self.inputs()
        args["x_level0_scale"] = [
            torch_npu.npu_format_cast(torch.ones((33, 3), device="npu"), 2)
        ]
        with self.assertRaises(RuntimeError):
            self.ops.dual_level_grouped_matmul(**args)

    def test_wrong_storage_dtype_rejected_by_bridge(self):
        for name, dtype, message in (
            ("x", torch.float32, "requires int8 or uint8 storage"),
            ("x_level0_scale", torch.float16, "requires float32 storage"),
        ):
            with self.subTest(name=name):
                args = self.inputs()
                tensor = args[name][0]
                args[name] = [torch_npu.npu_format_cast(tensor.to(dtype), 2)]
                with self.assertRaisesRegex(RuntimeError, name + ".*" + message):
                    self.ops.dual_level_grouped_matmul(**args)

    def test_transposed_scale_offset_rejected_by_bridge(self):
        args = self.inputs()
        args["weight"] = [
            torch_npu.npu_format_cast(
                torch.full((3, 32, 257), 0x22, dtype=torch.int8, device="npu"), 29
            ).transpose(1, 2)
        ]
        scale = torch_npu.npu_format_cast(
            torch.full((4, 32, 9, 2), 127, dtype=torch.int8, device="npu"), 2
        )
        args["weight_level1_scale"] = [scale[1:].transpose(1, 2)]
        with self.assertRaisesRegex(RuntimeError, "zero storage offset"):
            self.ops.dual_level_grouped_matmul(**args)

    def test_default_determinism(self):
        for transpose in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(transpose=transpose, dtype=dtype):
                    args = self.inputs()
                    args["output_dtype"] = dtype
                    # Keep non-unit scales fixed across calls; no determinism switch is set.
                    for name in ("x_level0_scale", "weight_level0_scale", "bias"):
                        original = args[name][0]
                        args[name] = [
                            torch_npu.npu_format_cast(
                                torch.rand_like(original) + 0.25, 2
                            )
                        ]
                    if transpose:
                        args["weight"] = [
                            torch_npu.npu_format_cast(
                                torch.full(
                                    (3, 32, 257), 0x22, dtype=torch.int8, device="npu"
                                ),
                                29,
                            ).transpose(1, 2)
                        ]
                        scale = (
                            args["weight_level1_scale"][0].transpose(1, 2).contiguous()
                        )
                        args["weight_level1_scale"] = [
                            torch_npu.npu_format_cast(scale, 2).transpose(1, 2)
                        ]
                    baseline = (
                        self.ops.dual_level_grouped_matmul(**args)[0]
                        .cpu()
                        .view(torch.int16)
                    )
                    for _ in range(5):
                        actual = (
                            self.ops.dual_level_grouped_matmul(**args)[0]
                            .cpu()
                            .view(torch.int16)
                        )
                        self.assertTrue(
                            torch.equal(actual, baseline),
                            "Repeated output differs at the bit level",
                        )

    def test_private_format_is_rejected_for_each_nd_list(self):
        for name in (
            "x",
            "x_level0_scale",
            "x_level1_scale",
            "weight_level0_scale",
            "weight_level1_scale",
            "bias",
        ):
            with self.subTest(name=name):
                args = self.inputs()
                original = args[name][0]
                converted = torch_npu.npu_format_cast(original, 29)
                self.assertEqual(converted.shape, original.shape)
                self.assertEqual(converted.stride(), original.stride())
                args[name] = [converted]
                with self.assertRaisesRegex(
                    RuntimeError, name + r".*format=.*expected ND"
                ):
                    self.ops.dual_level_grouped_matmul(**args)


if __name__ == "__main__":
    unittest.main()
