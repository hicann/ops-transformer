# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the production buffer-size helpers without loading Torch/NPU extensions.

Run with: python -m unittest discover -s mc2/common/tests/ut -p test_moe_ep_window_layout.py
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_window_helpers():
    source = (
        Path(__file__).resolve().parents[2] / "torch_extension" / "elastic_buffer.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    helpers = {
        "_inline_align",
        "_check_moe_comm_alg",
        "_MoeEpWindowLayout",
        "_get_moe_ep_window_layout",
        "_get_moe_ep_direct_window_bytes",
        "_get_moe_ep_window_bytes",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in helpers:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) and target.id.startswith("_MOE_")
            for target in node.targets
        ):
            nodes.append(node)

    def check(condition, message):
        if not condition:
            raise RuntimeError(message())

    scope = {"dataclass": dataclass, "torch": SimpleNamespace(_check=check)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


HELPERS = load_window_helpers()


class MoeEpWindowLayoutTest(unittest.TestCase):
    def layout(self, ep, algorithm):
        return HELPERS["_get_moe_ep_window_layout"](ep, 8, 128, ep * 2, 2, algorithm)

    def test_auto_preserves_legacy_window_layout(self):
        # These byte counts are regression anchors for the existing legacy ABI.
        for ep, state_bytes in [(2, 172544), (8, 203264), (15, 206336)]:
            with self.subTest(ep=ep):
                self.assertEqual(
                    self.layout(ep, "auto").state_buffer_bytes, state_bytes
                )

    def test_clos_below_16_preserves_legacy_window_layout(self):
        for ep in [2, 8, 15]:
            with self.subTest(ep=ep):
                self.assertEqual(self.layout(ep, "clos_only"), self.layout(ep, "auto"))

    def test_shared_layout_starts_at_16(self):
        for ep, state_bytes in [
            (16, 682496),
            (32, 1223168),
            (64, 2304512),
        ]:
            with self.subTest(ep=ep):
                clos = self.layout(ep, "clos_only")
                self.assertEqual(clos.state_buffer_bytes, state_bytes)
                self.assertNotEqual(
                    clos.state_buffer_bytes, self.layout(ep, "auto").state_buffer_bytes
                )

    def test_auto_16_rank_regression(self):
        self.assertEqual(self.layout(16, "auto").state_buffer_bytes, 200704)

    def test_comm_alg_rejects_empty_and_unknown_values(self):
        for algorithm in ["", "shared_jetty", "fullmesh", None]:
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(RuntimeError, "comm_alg must be"):
                    self.layout(16, algorithm)

    def test_clos_buffer_hint_covers_direct_window_and_is_2mb_aligned(self):
        for ep in [2, 8, 15, 16, 32, 64]:
            with self.subTest(ep=ep):
                required = HELPERS["_get_moe_ep_direct_window_bytes"](
                    self.layout(ep, "clos_only"), 8, 2
                )
                allocated = HELPERS["_get_moe_ep_window_bytes"](
                    ep, 8, 128, ep * 2, 2, "clos_only"
                )
                self.assertGreaterEqual(allocated, required)
                self.assertLess(allocated - required, 2 * 1024 * 1024)
                self.assertEqual(allocated % (2 * 1024 * 1024), 0)


if __name__ == "__main__":
    unittest.main()
