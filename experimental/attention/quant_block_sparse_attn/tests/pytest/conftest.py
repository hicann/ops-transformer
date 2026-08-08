#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import os
import sys
from pathlib import Path

import pytest


def _prepend_env_path(env_name, paths):
    current = os.environ.get(env_name, "")
    current_parts = [part for part in current.split(":") if part]
    new_parts = [str(path) for path in paths if path.exists() and str(path) not in current_parts]
    if new_parts:
        os.environ[env_name] = ":".join(new_parts + current_parts)


def _append_env_path(env_name, paths):
    current = os.environ.get(env_name, "")
    current_parts = [part for part in current.split(":") if part]
    new_parts = [str(path) for path in paths if path.exists() and str(path) not in current_parts]
    if new_parts:
        os.environ[env_name] = ":".join(current_parts + new_parts)


def _runtime_library_dirs():
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")
    candidates.extend([
        Path(sys.executable).resolve().parents[1] / "lib",
    ])
    deduped = []
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _preload_runtime_libraries():
    lib_dirs = _runtime_library_dirs()
    _prepend_env_path("LD_LIBRARY_PATH", lib_dirs)
    for lib_dir in lib_dirs:
        libstdcxx = lib_dir / "libstdc++.so.6"
        libgcc = lib_dir / "libgcc_s.so.1"
        if not libstdcxx.exists():
            continue
        if libgcc.exists():
            ctypes.CDLL(str(libgcc), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)
        return


def _custom_opp_roots():
    repo_root = Path(__file__).resolve().parents[5]
    build_candidates = [
        repo_root / "build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-x86_64.run"
        / "packages/vendors/custom_transformer",
    ]
    build_roots = [path for path in build_candidates if path.exists()]
    if build_roots:
        return build_roots
    candidates = [
        Path("/workspace/vendors/custom_transformer"),
    ]
    return [path for path in candidates if path.exists()]


def _configure_custom_opp_paths():
    custom_opp_roots = _custom_opp_roots()
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_roots)
    _prepend_env_path("LD_LIBRARY_PATH", [root / "op_api" / "lib" for root in custom_opp_roots])


_preload_runtime_libraries()
_configure_custom_opp_paths()


def pytest_addoption(parser):
    parser.addoption(
        "--check-accuracy",
        action="store_true",
        default=False,
        help="Enable detailed accuracy comparison including cosine similarity and error tables",
    )
    parser.addoption(
        "--force-regenerate",
        action="store_true",
        default=False,
        help="Force regenerate .pt test case files even if they already exist",
    )


@pytest.fixture(scope="session")
def check_accuracy(request):
    """Fixture that returns True when --check-accuracy flag is set."""
    return request.config.getoption("--check-accuracy")


@pytest.fixture(scope="session")
def force_regenerate(request):
    """Fixture that returns True when --force-regenerate flag is set."""
    return request.config.getoption("--force-regenerate")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "check_accuracy: mark test to run with detailed accuracy comparison",
    )
