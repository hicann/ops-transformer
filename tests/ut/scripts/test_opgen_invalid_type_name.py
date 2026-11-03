#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Use full opgen templates and actual SDK OpDef compilation/op_build."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/opgen/opgen_standalone.py"


class OpgenTypeNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sdk_path = os.environ.get("CANN_OPGEN_NATIVE_SDK") or os.environ.get(
            "ASCEND_HOME_PATH"
        )
        if not sdk_path:
            raise RuntimeError("Set CANN_OPGEN_NATIVE_SDK to an actual CANN SDK")
        cls.sdk = Path(sdk_path).resolve()
        configured_tool = os.environ.get("CANN_OPGEN_NATIVE_OPBUILD")
        candidates = (
            [Path(configured_tool)]
            if configured_tool
            else [
                cls.sdk / "tools/opbuild/op_build",
                cls.sdk.parent / "tools/opbuild/op_build",
            ]
        )
        cls.tool = next((p for p in candidates if p.is_file()), candidates[0]).resolve()
        if not (cls.sdk / "include/register/op_def_registry.h").is_file():
            raise RuntimeError("Complete native SDK OpDef headers are required")
        if not cls.tool.is_file():
            raise RuntimeError("Actual SDK op_build executable is required")

    def run_command(self, args, env=None):
        return subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def generate(self, name, output):
        return self.run_command(
            [
                "python3",
                str(SCRIPT),
                "--op_type",
                "math",
                "--op_name",
                name,
                "--output_path",
                str(output),
            ]
        )

    def compile_definition(self, source, library):
        return self.run_command(
            [
                "g++",
                "-std=c++11",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-fPIC",
                "-shared",
                str(source),
                "-I",
                str(self.sdk / "include"),
                "-L",
                str(self.sdk / "lib64"),
                "-lregister",
                "-Wl,-rpath," + str(self.sdk / "lib64"),
                "-o",
                str(library),
            ]
        )

    def test_invalid_derived_cpp_names_do_not_create_projects(self):
        for name in ["3d_add", "_9_value", "_", "___"]:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(prefix="opgen_invalid_type_") as temporary,
            ):
                output = Path(temporary)
                run = self.generate(name, output)
                messages = run.stdout + run.stderr
                if run.returncode == 0:
                    source = output / "math" / name / "op_host" / f"{name}_def.cpp"
                    compiled = self.compile_definition(source, output / "invalid.so")
                    self.assertNotEqual(compiled.returncode, 0)
                    messages += "\nNative generated OpDef compiler error:\n"
                    messages += compiled.stdout + compiled.stderr
                self.assertNotEqual(run.returncode, 0, messages)
                self.assertFalse((output / "math" / name).exists())

    def test_valid_generated_definitions_compile_and_emit_sdk_artifacts(self):
        for name, cpp_name in [
            ("native_add", "NativeAdd"),
            ("add_3d", "Add3d"),
            ("_custom_add", "CustomAdd"),
            ("add9", "Add9"),
        ]:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(prefix="opgen_native_type_") as temporary,
            ):
                output = Path(temporary)
                run = self.generate(name, output)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                source = output / "math" / name / "op_host" / f"{name}_def.cpp"
                library = output / "definition.so"
                compiled = self.compile_definition(source, library)
                self.assertEqual(
                    compiled.returncode, 0, compiled.stdout + compiled.stderr
                )
                emitted = output / "sdk-output"
                emitted.mkdir()
                env = os.environ.copy()
                env.update(
                    LD_LIBRARY_PATH=str(self.sdk / "lib64")
                    + ":"
                    + env.get("LD_LIBRARY_PATH", ""),
                    OPS_PROTO_SEPARATE="1",
                    OPS_ACLNN_GEN="1",
                    OPS_PROJECT_NAME="aclnn",
                )
                produced = self.run_command(
                    [str(self.tool), str(library), str(emitted)], env=env
                )
                self.assertEqual(
                    produced.returncode, 0, produced.stdout + produced.stderr
                )
                protos = list(emitted.glob("*_proto.h"))
                self.assertEqual(len(protos), 1)
                self.assertIn(f"REG_OP({cpp_name})", protos[0].read_text())
                self.assertTrue(list(emitted.glob("aclnn_*.h")))
                self.assertTrue(list(emitted.glob("aclnn_*.cpp")))
                for soc in ["ascend910b", "ascend910_93"]:
                    info = emitted / f"aic-{soc}-ops-info.ini"
                    self.assertTrue(info.is_file())
                    self.assertIn(f"[{cpp_name}]", info.read_text())


if __name__ == "__main__":
    unittest.main()
