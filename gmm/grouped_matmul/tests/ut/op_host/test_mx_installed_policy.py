# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class InstalledPolicyTest(unittest.TestCase):
    def test_buffered_policy_compiles_from_installed_kernel_tree(self):
        compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("A C++ compiler is required")
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / "ascendc/grouped_matmul/op_kernel"
            # Match the source-package rule: only op_kernel is installed.
            shutil.copytree(root / "op_kernel", installed)
            policy = (
                installed
                / "arch35/quant_adaptive_sliding_window_templates/gmm_weight_quant_mx_buffered_policy.h"
            )
            result = subprocess.run(
                [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++", "-"],
                input=f'#include "{policy.as_posix()}"\n',
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_kernel_headers_resolve_shared_constants_inside_package(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / "ascendc/grouped_matmul/op_kernel"
            shutil.copytree(root / "op_kernel", installed)
            for name in [
                "gmm_weight_quant_mx_buffered_policy.h",
                "gmm_weight_quant_tensor_api_mx_kernel.h",
            ]:
                header = (
                    installed / "arch35/quant_adaptive_sliding_window_templates" / name
                )
                include = next(
                    line.split('"')[1]
                    for line in header.read_text(encoding="utf-8").splitlines()
                    if line.startswith("#include")
                    and "mx_a8w4_pipeline_common.h" in line
                )
                resolved = (header.parent / include).resolve()
                self.assertTrue(resolved.is_file(), str(resolved))
                self.assertTrue(resolved.is_relative_to(installed.resolve()))


if __name__ == "__main__":
    unittest.main()
