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
"""
PyPTO kernel codegen driver.

This script is invoked from CMake configure stage for operators marked with enable_pypto_kernel(<op_file>).
It runs host-side binary artifact preparation and copies all generated artifacts into --out-dir.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock


def _install_build_only_torch_stub():
    """Stub runtime-only dependencies that binary codegen does not use."""
    sys.modules["torch"] = MagicMock(name="torch")
    sys.modules["torch_npu"] = MagicMock(name="torch_npu")


def main():
    parser = argparse.ArgumentParser(description="PyPTO kernel codegen driver")
    parser.add_argument("--py-file", required=True, help="kernel python source file")
    parser.add_argument(
        "--out-dir", required=True, help="directory to place generated artifacts"
    )
    parser.add_argument(
        "--op-file", required=True, help="op file stem (e.g. flash_attention_score_apt)"
    )
    args = parser.parse_args()

    py_file = Path(args.py_file).resolve()
    if not py_file.is_file():
        raise SystemExit(f"[pypto_codegen] py-file not found: {py_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _install_build_only_torch_stub()
    from pypto_pro.runtime.opc import prepare_binary_headers

    binary_dir = Path(prepare_binary_headers(str(py_file))).resolve()
    shutil.copytree(binary_dir, out_dir, dirs_exist_ok=True)
    copied = sorted(
        path.relative_to(binary_dir).as_posix()
        for path in binary_dir.rglob("*")
        if path.is_file()
    )

    logging.info("%s: generated %s into %s", args.op_file, copied, out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
