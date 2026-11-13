# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check public mode dispatch without an NPU or JIT compilation."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch


def load_implementation():
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
        Path(__file__).resolve().parents[2] / "torch_extension/indexer_quant_cache.py"
    )
    spec = importlib.util.spec_from_file_location("indexer_mode_test_module", source)
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(sys.modules, {"cann_ops_transformer.op_builder": builder_module}),
        patch("torch.library.impl", register),
    ):
        spec.loader.exec_module(module)
    return module, implementations


class IndexerQuantCacheModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.implementations = load_implementation()

    def test_mode_numbers_remain_compatible(self):
        for number, name in enumerate(
            ("mxfp8", "fp8", "hifloat8", "mxfp4", "hifloat4")
        ):
            with self.subTest(name=name):
                self.assertEqual(self.module._resolve_quant_mode(name), number)
                self.assertEqual(self.module._resolve_quant_mode(number), number)
        self.assertEqual(self.module._resolve_quant_mode(" HiFloat4 "), 4)

    def test_invalid_mode_rejected(self):
        for mode in (5, -1, "hifloat5"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.module._resolve_quant_mode(mode)

    def test_hifloat4_dispatch_preserves_inplace_carriers(self):
        cache = torch.empty((2, 16, 1, 64), dtype=torch.uint8)
        scale = torch.empty((2, 16, 1, 2), dtype=torch.float32)
        x = torch.empty((1, 128), dtype=torch.float16)
        slots = torch.zeros(1, dtype=torch.int32)
        extension = types.SimpleNamespace(indexer_quant_cache=Mock(return_value=None))
        with patch.object(
            self.module.indexer_quant_cache_op_builder,
            "load",
            return_value=extension,
            create=True,
        ):
            result = self.implementations["PrivateUse1"](
                cache,
                scale,
                x,
                slots,
                quant_mode="hifloat4",
                round_scale=False,
                x_scale=2.0,
            )
        self.assertIsNone(result)
        actual = extension.indexer_quant_cache.call_args.args
        for received, original in zip(actual[:4], (cache, scale, x, slots)):
            self.assertIs(received, original)
        self.assertEqual(actual[4:], (4, False, 2.0))


if __name__ == "__main__":
    unittest.main()
