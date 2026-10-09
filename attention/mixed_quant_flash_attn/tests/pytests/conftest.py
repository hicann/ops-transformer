#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys

import pytest

_PYT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PYT_DIR not in sys.path:
    sys.path.insert(0, _PYT_DIR)


def pytest_addoption(parser):
    parser.addoption(
        "--mode",
        nargs="+",
        default=["gen"],
        help="Test mode: gen, cpu, npu, compare",
    )
    parser.addoption(
        "--testset",
        action="store",
        default="functional_stc",
        help="Test data directory name (e.g. functional_stc, functional_rdv, functional_redline)",
    )
    parser.addoption(
        "--case_id",
        action="store",
        default="all",
        help="Case ID filter: 'all' or comma-separated names",
    )
    parser.addoption(
        "--device_id",
        action="store",
        default=0,
        type=int,
        help="NPU device ID",
    )
    parser.addoption(
        "--seed",
        action="store",
        default=21,
        type=int,
        help="Random seed for data generation",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ci: mark a test as a CI test")
    config.addinivalue_line(
        "markers", "func_rdv: mark a test as a functional correctness RDV test"
    )
    config.addinivalue_line(
        "markers", "func_redline: mark a test as a functional redline test"
    )
    config.addinivalue_line("markers", "func_stc: mark a test as a functional STC test")
    config.addinivalue_line(
        "markers", "func_test: mark a test as a functional TEST test"
    )
    config.addinivalue_line(
        "markers", "perf_rdv: mark a test as a performance RDV test"
    )
    config.addinivalue_line(
        "markers", "perf_redline: mark a test as a performance redline test"
    )
    config.addinivalue_line("markers", "debug: mark a test as a daily debug test")


@pytest.fixture(scope="session")
def device_id(request):
    return request.config.getoption("--device_id")


@pytest.fixture(scope="session")
def seed(request):
    return request.config.getoption("--seed")
