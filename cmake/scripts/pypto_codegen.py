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
import importlib.util
import logging
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock


def _install_build_only_torch_stub():
    """Stub runtime-only dependencies that binary codegen does not use."""
    sys.modules["torch"] = MagicMock(name="torch")
    sys.modules["torch_npu"] = MagicMock(name="torch_npu")


def _load_legacy_kernel(py_file: Path):
    """Load the sole @pl.jit kernel for the legacy header-generation API."""
    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location("_pypto_codegen_kernel_mod", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel module from {py_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    kernels = [
        value for value in vars(module).values() if isinstance(value, _TileJitKernel)
    ]
    if len(kernels) != 1:
        raise RuntimeError(
            f"kernel module '{py_file}' must define exactly one @pl.jit kernel, found {len(kernels)}"
        )
    return kernels[0]


def _prepare_binary_headers(py_file: Path) -> Path:
    try:
        from pypto_pro.runtime.opc import prepare_binary_headers

        return Path(prepare_binary_headers(str(py_file))).resolve()
    except Exception as error:
        logging.warning(
            "prepare_binary_headers failed, falling back to generate_binary_headers: %s",
            error,
        )

    import pypto_pro.runtime.opc.pypto_compile as pto_compile

    return Path(
        pto_compile.generate_binary_headers(_load_legacy_kernel(py_file))
    ).resolve()


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
    binary_dir = _prepare_binary_headers(py_file)
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
