# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise the registered Meta implementation without loading an NPU extension."""

import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch


def load_implementations():
    implementations = {}

    class Builder:
        def __init__(self, name, **kwargs):
            self.name = name

        def _ensure_initialized(self):
            self.register_meta()

    def register(library, name, dispatch):
        def decorator(function):
            implementations[dispatch] = function
            return function

        return decorator

    builder_module = types.ModuleType("cann_ops_transformer.op_builder")
    builder_module.OpBuilder = Builder
    builder_module.get_as_library = lambda: None
    source = (
        Path(__file__).resolve().parents[2]
        / "torch_extension/grouped_matmul_activation.py"
    )
    spec = importlib.util.spec_from_file_location("activation_meta_test_module", source)
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, {"cann_ops_transformer.op_builder": builder_module}),
        patch("torch.library.impl", register),
    ):
        spec.loader.exec_module(module)
    return module, implementations


class ActivationMeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, implementations = load_implementations()
        cls.meta = staticmethod(implementations["Meta"])
        cls.execute = staticmethod(implementations["PrivateUse1"])

    def inputs(self, transposed=False):
        weight = torch.empty((2, 128, 64), dtype=torch.int8, device="meta")
        scale = torch.empty((2, 2, 128, 2), dtype=torch.uint8, device="meta")
        if transposed:
            weight = torch.empty(
                (2, 128, 64), dtype=torch.int8, device="meta"
            ).transpose(1, 2)
            scale = torch.empty(
                (2, 128, 2, 2), dtype=torch.uint8, device="meta"
            ).permute(0, 2, 1, 3)
        return [
            torch.empty((3, 64), dtype=torch.int8, device="meta"),
            torch.empty((3, 2, 2), dtype=torch.uint8, device="meta"),
            torch.empty((2,), dtype=torch.int64, device="meta"),
            [weight],
            [scale],
        ]

    def test_legal_layouts_and_group_types(self):
        for transposed in (False, True):
            for group_type in (0, 1):
                with self.subTest(transposed=transposed, group_type=group_type):
                    result = self.meta(
                        *self.inputs(transposed), group_list_type=group_type
                    )
                    self.assertEqual(result.shape, (3, 64))
                    self.assertEqual(result.dtype, torch.bfloat16)

    def test_public_signature_and_schema(self):
        schema = torch._C.parse_schema(self.module._op_builder.schema())
        expected = [
            "x",
            "x_scale",
            "group_list",
            "weight",
            "weight_scale",
            "bias",
            "activation_type",
            "group_list_type",
            "tuning_config",
            "swiglu_mode",
            "clamp_limit",
            "glu_alpha",
            "glu_bias",
            "x_dtype",
            "weight_dtype",
            "x_scale_dtype",
            "weight_scale_dtype",
        ]
        self.assertEqual([argument.name for argument in schema.arguments], expected)
        for function in (
            self.module.grouped_matmul_activation,
            self.meta,
            self.execute,
        ):
            self.assertEqual(list(inspect.signature(function).parameters), expected)
        for removed in ("dequant_mode", "dequant_dtype", "act_type"):
            with self.subTest(parameter=removed), self.assertRaises(TypeError):
                self.meta(*self.inputs(), **{removed: 0})

    def test_activation_type(self):
        self.meta(*self.inputs(), activation_type="swiglu")
        with self.assertRaisesRegex(ValueError, "activation_type"):
            self.meta(*self.inputs(), activation_type="relu")

    def test_execute_argument_order(self):
        extension = types.SimpleNamespace(
            grouped_matmul_activation=Mock(return_value="result")
        )
        args = self.inputs()
        with patch.object(
            self.module._op_builder, "load", return_value=extension, create=True
        ):
            result = self.execute(
                *args,
                activation_type="swiglu",
                group_list_type=1,
                tuning_config=[],
                clamp_limit=5.0,
                glu_alpha=1.5,
                glu_bias=0.5,
            )
        self.assertEqual(result, "result")
        extension.grouped_matmul_activation.assert_called_once_with(
            *args,
            None,
            "swiglu",
            1,
            [],
            self.module.SWIGLU_MODE_SPLIT,
            5.0,
            1.5,
            0.5,
            None,
            None,
            None,
            None,
        )

    def test_dtype_overrides(self):
        for transposed in (False, True):
            for offset in (0, self.module.ACL_DTYPE_OFFSET):
                with self.subTest(transposed=transposed, offset=offset):
                    result = self.meta(
                        *self.inputs(transposed),
                        x_dtype=40 + offset,
                        weight_dtype=40 + offset,
                        x_scale_dtype=37 + offset,
                        weight_scale_dtype=37 + offset,
                    )
                    self.assertEqual(result.shape, (3, 64))
        for name in ("x_dtype", "weight_dtype", "x_scale_dtype", "weight_scale_dtype"):
            for value in (41, 297, torch.float32, True, "fp4"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        self.meta(*self.inputs(), **{name: value})

    def test_native_dtype_storage(self):
        for transposed in (False, True):
            args = self.inputs(transposed)
            fp4 = getattr(torch, "float4_e2m1fn_x2", None)
            e8m0 = getattr(torch, "float8_e8m0fnu", None)
            if fp4 is None or e8m0 is None:
                self.skipTest("native FP4/E8M0 dtypes are unavailable")
            args[0] = args[0].view(fp4)
            args[1] = args[1].view(e8m0)
            args[3] = [args[3][0].view(fp4)]
            args[4] = [args[4][0].view(e8m0)]
            result = self.meta(
                *args,
                x_dtype=fp4,
                weight_dtype=fp4,
                x_scale_dtype=e8m0,
                weight_scale_dtype=e8m0,
            )
            self.assertEqual(result.shape, (3, 64))

    def test_dtype_forwarding(self):
        args = self.inputs()
        with patch.object(
            torch.ops.cann_ops_transformer, "grouped_matmul_activation", create=True
        ) as operation:
            self.module.grouped_matmul_activation(
                *args,
                x_dtype=40,
                weight_dtype=296,
                x_scale_dtype=37,
                weight_scale_dtype=293,
            )
        for name, value in (
            ("x_dtype", 296),
            ("weight_dtype", 296),
            ("x_scale_dtype", 293),
            ("weight_scale_dtype", 293),
        ):
            self.assertEqual(operation.call_args.kwargs[name], value)
        extension = types.SimpleNamespace(grouped_matmul_activation=Mock())
        with patch.object(
            self.module._op_builder, "load", return_value=extension, create=True
        ):
            self.execute(
                *args,
                x_dtype=296,
                weight_dtype=296,
                x_scale_dtype=293,
                weight_scale_dtype=293,
            )
        self.assertEqual(
            extension.grouped_matmul_activation.call_args.args[-4:],
            (296, 296, 293, 293),
        )

    def test_public_keyword_forwarding(self):
        args = self.inputs()
        with patch.object(
            torch.ops.cann_ops_transformer, "grouped_matmul_activation", create=True
        ) as operation:
            self.module.grouped_matmul_activation(*args, activation_type="swiglu")
        self.assertEqual(operation.call_args.kwargs["activation_type"], "swiglu")
        for removed in ("dequant_mode", "dequant_dtype", "act_type"):
            self.assertNotIn(removed, operation.call_args.kwargs)

    def test_float32_attribute_bounds(self):
        for attribute in ("clamp_limit", "glu_alpha", "glu_bias"):
            for value in (math.inf, math.nan, 1e100, -1e100):
                with self.subTest(attribute=attribute, value=value):
                    with self.assertRaisesRegex(ValueError, "float32"):
                        self.meta(*self.inputs(), **{attribute: value})
        for value in (0.0, -1.0, 1e-50):
            with self.subTest(clamp_limit=value):
                with self.assertRaisesRegex(ValueError, "positive"):
                    self.meta(*self.inputs(), clamp_limit=value)
        self.meta(*self.inputs(), clamp_limit=2.0**-149, glu_alpha=-1.0, glu_bias=0.0)

    def test_packed_input_dtypes(self):
        for index in (0, 1, 3, 4):
            args = self.inputs()
            if index >= 3:
                args[index] = [args[index][0].float()]
            else:
                args[index] = args[index].float()
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "storage"),
            ):
                self.meta(*args)
        args = self.inputs()
        args[2] = args[2].int()
        with self.assertRaisesRegex(ValueError, "int64"):
            self.meta(*args)

    def test_shape_constraints(self):
        cases = [
            (0, torch.empty((3, 65), dtype=torch.int8, device="meta"), "logical K"),
            (1, torch.empty((3, 1, 2), dtype=torch.uint8, device="meta"), "x_scale"),
            (
                3,
                [torch.empty((2, 256, 64), dtype=torch.int8, device="meta")],
                "weight\\[0\\]",
            ),
            (
                4,
                [torch.empty((2, 2, 192, 2), dtype=torch.uint8, device="meta")],
                "logical N",
            ),
            (
                4,
                [torch.empty((1, 2, 128, 2), dtype=torch.uint8, device="meta")],
                "weight_scale",
            ),
        ]
        for index, tensor, message in cases:
            args = self.inputs()
            args[index] = tensor
            with (
                self.subTest(index=index, message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.meta(*args)
        args = self.inputs(True)
        args[4] = [args[4][0].contiguous()]
        with self.assertRaisesRegex(ValueError, "consistent transpose"):
            self.meta(*args)


if __name__ == "__main__":
    unittest.main()
