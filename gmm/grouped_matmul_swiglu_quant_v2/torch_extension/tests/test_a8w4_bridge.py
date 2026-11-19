# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU bridge contracts, without torch_npu/CANN. Run with unittest discover."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


EXTENSION = Path(__file__).resolve().parents[1]
ROOT = EXTENSION.parents[2]
FP4_ATTR = 296
E8M0_ATTR = 293


def load_bridge():
    registered = {}

    class Builder:
        def __init__(self, name, **kwargs):
            self.name = name

        def _ensure_initialized(self):
            pass

    builder_module = types.ModuleType("cann_ops_transformer.op_builder")
    builder_module.OpBuilder = Builder
    builder_module.get_as_library = lambda: None
    npu = types.ModuleType("torch_npu")
    npu.float4_e2m1fn_x2 = FP4_ATTR
    npu.float8_e8m0fnu = E8M0_ATTR

    def register(lib, name, dispatch):
        def decorate(function):
            registered[dispatch] = function
            return function

        return decorate

    spec = importlib.util.spec_from_file_location(
        "gmmsq_bridge_cpu", EXTENSION / "grouped_matmul_swiglu_quant.py"
    )
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.dict(
            sys.modules,
            {"torch_npu": npu, "cann_ops_transformer.op_builder": builder_module},
        ),
        mock.patch("torch.library.impl", register),
    ):
        spec.loader.exec_module(module)
        module._op_builder.register_meta()
    return module, registered["Meta"]


class TestA8W4Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge, meta = load_bridge()
        cls.meta = staticmethod(meta)

    def inputs(self, n=128, nk=False, view=True, native=False):
        scale_shape = (4, n, 2, 2) if nk else (4, 2, n, 2)
        scale = torch.empty(scale_shape, dtype=torch.uint8)
        if view:
            scale = scale.transpose(1, 2).contiguous().transpose(1, 2)
        weight = torch.empty(
            (4, n, 128 // 2),
            dtype=torch.float4_e2m1fn_x2 if native else torch.uint8,
        ).transpose(-1, -2)
        return (
            torch.empty((17, 128), dtype=torch.float8_e4m3fn),
            [weight],
            [scale],
            torch.empty((17, 2, 2), dtype=torch.uint8),
            torch.empty(4, dtype=torch.int64),
        )

    def test_dtype_normalization(self):
        bridge = self.bridge
        for dtype in (torch.float4_e2m1fn_x2, FP4_ATTR):
            self.assertEqual(bridge._normalize_wrapper_dtype(dtype), FP4_ATTR)
        for dtype in (6, torch.float32, None):
            self.assertEqual(
                bridge._normalize_dtype(dtype),
                6 if dtype == 6 else (0 if dtype is torch.float32 else None),
            )

    def test_meta_transposed_scale_and_smooth(self):
        for n in (128, 256, 384):
            for nk in (False,):
                for view in (True,):
                    for native in (False, True):
                        for smooth in (None, torch.ones((4, n // 2))):
                            with self.subTest(n=n, nk=nk, view=view, native=native):
                                y, scale = self.meta(
                                    *self.inputs(n, nk, view, native),
                                    swiglu_mode=2,
                                    weight_dtype=None if native else FP4_ATTR,
                                    smooth_scale=smooth,
                                )
                                self.assertEqual(tuple(y.shape), (17, n // 2))
                                self.assertEqual(
                                    tuple(scale.shape), (17, (n + 127) // 128, 2)
                                )
                                self.assertEqual(y.dtype, torch.float8_e4m3fn)

    def test_meta_rejects_invalid_a8w4_smooth(self):
        for smooth in (
            torch.ones(4),
            torch.ones((4, 64), dtype=torch.float16),
            torch.ones((4, 128)),
            torch.ones((3, 64)),
        ):
            with self.subTest(shape=smooth.shape), self.assertRaises(RuntimeError):
                self.meta(
                    *self.inputs(),
                    swiglu_mode=2,
                    weight_dtype=FP4_ATTR,
                    smooth_scale=smooth,
                )

    def test_meta_rejects_non_transposed_weight_or_scale(self):
        for target in ("weight", "weight_scale", "both"):
            x, weight, scales, xs, groups = self.inputs()
            if target in ("weight", "both"):
                weight = [weight[0].contiguous()]
            if target in ("weight_scale", "both"):
                scales = [scales[0].contiguous()]
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(RuntimeError, "transposed WeightNZ"),
            ):
                self.meta(
                    x, weight, scales, xs, groups, swiglu_mode=2, weight_dtype=FP4_ATTR
                )

    def test_meta_smooth_device_matches_x(self):
        x, weight, scales, xs, groups = self.inputs()
        for x_device, smooth_device in (("cpu", "meta"), ("meta", "cpu")):
            with self.subTest(x=x_device, smooth=smooth_device):
                with self.assertRaisesRegex(RuntimeError, "same device"):
                    self.meta(
                        x.to(x_device),
                        weight,
                        scales,
                        xs,
                        groups,
                        swiglu_mode=2,
                        weight_dtype=FP4_ATTR,
                        smooth_scale=torch.ones((4, 64), device=smooth_device),
                    )
        y, _ = self.meta(
            x.to("meta"),
            weight,
            scales,
            xs,
            groups,
            swiglu_mode=2,
            weight_dtype=FP4_ATTR,
            smooth_scale=torch.ones((4, 64), device="meta"),
        )
        self.assertEqual(tuple(y.shape), (17, 64))

    def test_missing_k_axis_is_rejected(self):
        x, weight, _, xs, groups = self.inputs()
        with self.assertRaisesRegex(RuntimeError, "ceil"):
            self.meta(
                x,
                weight,
                [torch.empty((4, 3, 128, 2))],
                xs,
                groups,
                swiglu_mode=2,
                weight_dtype=FP4_ATTR,
            )

    def test_fake_tensor_a8w4(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        with FakeTensorMode():
            for n in (128, 384):
                for nk in (False,):
                    for view in (True,):
                        y, scale = self.meta(
                            *self.inputs(n, nk, view),
                            swiglu_mode=2,
                            weight_dtype=FP4_ATTR,
                            smooth_scale=torch.ones((4, n // 2)),
                        )
                        self.assertIsInstance(y, FakeTensor)
                        self.assertIsInstance(scale, FakeTensor)
                        self.assertEqual(tuple(y.shape), (17, n // 2))

    def test_old_meta_and_defaults_are_unchanged(self):
        x, _, scales, xs, groups = self.inputs()
        weight = [torch.empty((4, 128, 128), dtype=torch.float8_e4m3fn)]
        # A placeholder smooth was not checked by old Meta; do not tighten it.
        y, scale = self.meta(
            x, weight, scales, xs, groups, swiglu_mode=2, smooth_scale=torch.empty(0)
        )
        self.assertEqual(tuple(y.shape), (17, 64))
        self.assertEqual(tuple(scale.shape), (17, 1, 2))
        with self.assertRaisesRegex(ValueError, "requires swiglu_mode=2"):
            self.meta(x, weight, scales, xs, groups)
        schema = torch._C.parse_schema(self.bridge._op_builder.schema())
        defaults = {arg.name: arg.default_value for arg in schema.arguments}
        self.assertIsNone(defaults["swiglu_mode"])
        self.assertEqual(defaults["dequant_mode"], 2)
        self.assertEqual(defaults["dequant_dtype"], 6)
        self.assertEqual(defaults["quant_mode"], 2)

    def test_public_call_forwards_device_inputs_and_attrs(self):
        inputs = self.inputs()
        smooth = torch.ones((4, 64))
        operation = mock.Mock(return_value=("y", "scale"))
        with mock.patch.object(
            torch.ops.cann_ops_transformer,
            "grouped_matmul_swiglu_quant",
            operation,
            create=True,
        ):
            result = self.bridge.grouped_matmul_swiglu_quant(
                *inputs,
                swiglu_mode=2,
                weight_dtype=FP4_ATTR,
                smooth_scale=smooth,
                clamp_limit=3.0,
                glu_alpha=0.5,
                glu_bias=-1.0,
            )
        self.assertEqual(result, ("y", "scale"))
        args, kwargs = operation.call_args
        self.assertIs(args[4], inputs[4])
        self.assertIs(kwargs["smooth_scale"], smooth)
        self.assertEqual(kwargs["weight_dtype"], FP4_ATTR)
        self.assertEqual(kwargs["dequant_dtype"], 6)
        self.assertEqual(kwargs["clamp_limit"], 3.0)
        self.assertEqual(kwargs["glu_alpha"], 0.5)
        self.assertEqual(kwargs["glu_bias"], -1.0)

    def test_compiled_cpp_route(self):
        text = (EXTENSION / "csrc/grouped_matmul_swiglu_quant.cpp").read_text(
            encoding="utf-8"
        )
        source = text[text.index("namespace op_api {") : text.index("PYBIND11_MODULE(")]
        mock_header = Path(__file__).with_name("a8w4_bridge_mock.h")
        main = (
            Path(__file__).with_name("a8w4_bridge_main.cpp").read_text(encoding="utf-8")
        )
        self.compile_and_run(
            f'#include "{mock_header.as_posix()}"\n' + source + "}\n" + main
        )

    def test_real_packed_shape_converter(self):
        header = (
            ROOT / "torch_extension/cann_ops_transformer/common/aclnn_common.h"
        ).read_text(encoding="utf-8")
        start = header.index("inline void CollectB4ShapeInfo(")
        helper = header[start : header.index("enum QuantMode", start)]
        mock_header = Path(__file__).with_name("a8w4_bridge_mock.h")
        source = f'#include "{mock_header.as_posix()}"\n' + helper + "\nint main() {\n"
        for n in (128, 384):
            for dtype in (torch.uint8, torch.float4_e2m1fn_x2):
                for shape in ((4, 128, n), (4, n, 128)):
                    packed = torch.empty((*shape[:-1], shape[-1] // 2), dtype=dtype)
                    logical = torch.empty(shape, device="meta")
                    self.assertEqual(packed.element_size(), 1)
                    self.assertEqual(
                        packed.untyped_storage().nbytes() * 2, logical.numel()
                    )
                    for value, expected in (
                        (packed, logical),
                        (packed.transpose(1, 2), logical.transpose(1, 2)),
                    ):
                        dims = ",".join(map(str, value.shape))
                        strides = ",".join(map(str, value.stride()))
                        expected_dims = ",".join(map(str, expected.shape))
                        expected_strides = ",".join(map(str, expected.stride()))
                        source += (
                            "{ at::Tensor tensor({"
                            + dims
                            + "}, at::ScalarType::Byte);\n"
                            "c10::SmallVector<int64_t, MAX_DIM_NUM> dims{"
                            + dims
                            + "}, strides{"
                            + strides
                            + "};\n"
                            "CollectB4ShapeInfo(tensor, strides, dims);\n"
                            "require(dims == std::vector<int64_t>{"
                            + expected_dims
                            + "});\n"
                            "require(strides == std::vector<int64_t>{"
                            + expected_strides
                            + "}); }\n"
                        )
        self.compile_and_run(source + "}\n")

    def compile_and_run(self, source):
        compiler = (
            os.environ.get("GMMSQ_TEST_CXX")
            or shutil.which("c++")
            or shutil.which("g++")
        )
        if not compiler:
            self.skipTest("Set GMMSQ_TEST_CXX to a C++17 compiler")
        with tempfile.TemporaryDirectory(prefix="gmmsq-extension-") as directory:
            path = Path(directory)
            (path / "test.cpp").write_text(source, encoding="utf-8")
            command = [compiler]
            result = subprocess.run(
                command
                + ["-std=c++17", str(path / "test.cpp"), "-o", str(path / "test.exe")],
                capture_output=True,
                text=True,
                errors="replace",
            )
            self.assertEqual(result.returncode, 0, result.stderr[-12000:])
            subprocess.run([str(path / "test.exe")], check=True)


if __name__ == "__main__":
    unittest.main()
