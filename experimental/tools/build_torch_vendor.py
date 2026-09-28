#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build selected experimental Torch adapters without modifying source files.

The repository's setup.py is run only in a temporary checkout containing the
Torch framework and selected adapters. Its vendor import compatibility fix is
applied to that copy, never to the repository's public OpBuilder.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


BASE_PACKAGE = "cann_ops_transformer"
ABSOLUTE_IMPORT = "from cann_ops_transformer.utils.arg_check import wrap_op_module"
RELATIVE_IMPORT = "from ..utils.arg_check import wrap_op_module"
COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", "build", "dist", "*.egg-info"
)


def identifier(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise argparse.ArgumentTypeError(f"Invalid Python identifier: {value!r}")
    return value


def find_adapters(repo, names):
    """Match only experimental operators, avoiding main-tree name collisions."""
    selected = []
    for name in names:
        matches = sorted((repo / "experimental").glob(f"*/{name}/torch_extension"))
        matches = [path for path in matches if path.is_dir()]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one experimental adapter for {name!r}, found {len(matches)}"
            )
        selected.append(matches[0])
    return selected


def build_vendor(repo, names, vendor, output_dir, install_dir=None):
    adapters = find_adapters(repo, names)
    framework = repo / "torch_extension"
    package = f"{BASE_PACKAGE}_{vendor}"
    output_dir = output_dir.resolve()
    if install_dir is not None:
        install_dir = install_dir.resolve()

    with tempfile.TemporaryDirectory(prefix="ops-transformer-vendor-") as temporary:
        checkout = Path(temporary) / "ops-transformer"
        copied_framework = checkout / "torch_extension"
        copied_framework.mkdir(parents=True)
        shutil.copy2(framework / "setup.py", copied_framework / "setup.py")
        shutil.copytree(
            framework / BASE_PACKAGE,
            copied_framework / BASE_PACKAGE,
            ignore=COPY_IGNORE,
        )
        for adapter in adapters:
            shutil.copytree(
                adapter, checkout / adapter.relative_to(repo), ignore=COPY_IGNORE
            )

        builder = copied_framework / BASE_PACKAGE / "op_builder/builder.py"
        contents = builder.read_text()
        if contents.count(ABSOLUTE_IMPORT) == 1:
            builder.write_text(contents.replace(ABSOLUTE_IMPORT, RELATIVE_IMPORT, 1))
        elif contents.count(RELATIVE_IMPORT) != 1:
            raise RuntimeError(
                "Unrecognized OpBuilder import; update the temporary vendor compatibility patch"
            )

        env = os.environ.copy()
        env["TORCH_EXTENSION_OPS"] = ",".join(names)
        env["TORCH_EXTENSION_VENDOR"] = vendor
        print(
            f"Building {package}; adapters={','.join(names)}; temporary checkout={checkout}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel"],
            cwd=copied_framework,
            env=env,
            check=True,
        )
        wheels = list((copied_framework / "dist").glob(f"{package}-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel for {package}, found {len(wheels)}")
        with zipfile.ZipFile(wheels[0]) as wheel:
            packaged_builder = wheel.read(f"{package}/op_builder/builder.py").decode()
            if (
                RELATIVE_IMPORT not in packaged_builder
                or ABSOLUTE_IMPORT in packaged_builder
            ):
                raise RuntimeError(
                    "Vendor wheel is missing its local import compatibility fix"
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        result = output_dir / wheels[0].name
        shutil.copy2(wheels[0], result)

    print(f"Wheel: {result}", flush=True)
    if install_dir is not None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--target",
                str(install_dir),
                str(result),
            ],
            check=True,
        )
        print(f"Installed {package} into {install_dir}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops",
        required=True,
        help="Comma-separated experimental operator directory names",
    )
    parser.add_argument("--vendor", required=True, type=identifier)
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to retain the wheel"
    )
    parser.add_argument(
        "--install-dir", type=Path, help="Optional pip --target installation directory"
    )
    args = parser.parse_args()
    try:
        names = list(
            dict.fromkeys(identifier(name.strip()) for name in args.ops.split(","))
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[2]
    build_vendor(repo, names, args.vendor, args.output_dir, args.install_dir)


if __name__ == "__main__":
    main()
