#!/usr/bin/python3
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

import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON_DIR = os.path.join(_HERE, "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)
# golden 模块在 assets/ 目录下
_ASSETS_DIR = os.path.join(_HERE, "..", "..", "assets")
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)
# golden 模块依赖 pytest/common/ 下的 result_compare_method
_OLD_COMMON_DIR = os.path.join(_HERE, "..", "common")
if _OLD_COMMON_DIR not in sys.path:
    sys.path.insert(0, _OLD_COMMON_DIR)

_VALID_MODES = {"all", "gen", "cpu", "npu", "compare"}


def _parse_golden_mode(raw):
    parts = {m.strip() for m in raw.split(",") if m.strip()}
    invalid = parts - _VALID_MODES
    if invalid:
        raise pytest.UsageError(
            f"Invalid golden-mode: {invalid}. Valid: {_VALID_MODES}"
        )
    if "all" in parts:
        return {"gen", "cpu", "npu", "compare"}
    return parts


def pytest_addoption(parser):
    parser.addoption(
        "--golden-mode",
        default="all",
        help="Golden 执行模式: all=全流程, gen=生成数据, cpu=跑CPU, npu=跑NPU, compare=精度对比",
    )
    parser.addoption(
        "--cache-dir",
        default=None,
        help="golden 缓存目录路径",
    )
    parser.addoption(
        "--device",
        default=0,
        type=int,
        metavar="DEVICE_ID",
        help="NPU 设备编号（默认 0）",
    )


@pytest.fixture(scope="session")
def golden_mode(request):
    return _parse_golden_mode(request.config.getoption("--golden-mode"))


@pytest.fixture(scope="session")
def cache_dir(request):
    return request.config.getoption("--cache-dir")
