# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile and execute small host tests against the production headers/methods."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def run_cpp(source):
    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
    if not compiler:
        raise unittest.SkipTest("Requires a C++17 host compiler")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "check.cpp"
        binary = Path(directory) / "check.exe"
        path.write_text(source, encoding="utf-8")
        if Path(compiler).stem.lower() == "cl":
            arguments = [
                compiler,
                "/nologo",
                "/std:c++17",
                "/EHsc",
                "/O2",
                str(path),
                "/Fe:" + str(binary),
                "/Fo:" + str(Path(directory) / "check.obj"),
            ]
        else:
            arguments = [compiler, "-std=c++17", "-O2", str(path), "-o", str(binary)]
        result = subprocess.run(
            arguments, capture_output=True, text=True, errors="replace", timeout=120
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        subprocess.run(
            [str(binary)],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
