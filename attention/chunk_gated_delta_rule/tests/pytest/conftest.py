# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)
import csv
import os
import pytest
import torch
import torch_npu  # noqa: F401

_RESULT_ROWS = []
_CURRENT_SEED = 0


def _get_check_type():
    """precision=带CPU golden精度对比；
    execution_only=仅NPU执行（random_npu模式）。"""
    if os.environ.get("SKIP_GOLDEN", "0") != "1":
        return "precision"
    return "execution_only"


@pytest.fixture(autouse=True)
def _set_random_seed():
    global _CURRENT_SEED
    fix_seed = os.environ.get("TORCH_SEED", "")
    if fix_seed:
        _CURRENT_SEED = int(fix_seed)
        torch.manual_seed(_CURRENT_SEED)
        torch.npu.manual_seed(_CURRENT_SEED)
    else:
        _CURRENT_SEED = torch.seed()
        torch.npu.manual_seed(_CURRENT_SEED)
    logger.info(f"[seed] {_CURRENT_SEED}")
    yield


def _get_model_name():
    if os.environ.get("USE_GRAPH", "false").lower() == "true":
        return "aclgraph"
    return "torch直调"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call":
        return

    params = item.funcargs.get("param_combinations", {})

    if report.passed:
        status = "PASSED"
    elif report.failed:
        status = "FAILED"
    elif report.skipped:
        status = "SKIPPED"
    else:
        status = "UNKNOWN"

    error = ""
    if status in ("FAILED", "ERROR") and report.longrepr:
        error = str(report.longreprtext).replace("\n", " | ")[:2000]

    row = {
        "random_seed": os.environ.get("RANDOM_SEED", ""),
        "seed": _CURRENT_SEED,
        "test_name": params.get("_name", ""),
        "test_mode": os.environ.get("TEST_MODE", ""),
        "check_type": _get_check_type(),
        "model": _get_model_name(),
        "status": status,
        "B": params.get("B", ""),
        "seqlen": params.get("seqlen", ""),
        "nk": params.get("nk", ""),
        "nv": params.get("nv", ""),
        "dk": params.get("dk", ""),
        "dv": params.get("dv", ""),
        "chunk_size": params.get("chunk_size", ""),
        "data_type": str(params.get("data_type", "")),
        "state_data_type": str(params.get("state_data_type", "")),
        "has_g": params.get("has_g", ""),
        "is_continue": params.get("is_contiguous", ""),
        "query_datarange": str(params.get("query_datarange", "")),
        "key_datarange": str(params.get("key_datarange", "")),
        "value_datarange": str(params.get("value_datarange", "")),
        "gamma_datarange": str(params.get("gamma_datarange", "")),
        "beta_datarange": str(params.get("beta_datarange", "")),
        "state_datarange": str(params.get("state_datarange", "")),
        "errmsg": error,
        "durations": "",
    }
    _RESULT_ROWS.append(row)


def pytest_runtest_logfinish(nodeid, location):
    logger.info("")


def pytest_sessionfinish(session, exitstatus):
    csv_file = os.environ.get("CSV_FILE", "")
    if not csv_file or not _RESULT_ROWS:
        return

    fields = [
        "random_seed",
        "seed",
        "test_name",
        "test_mode",
        "check_type",
        "model",
        "status",
        "B",
        "seqlen",
        "nk",
        "nv",
        "dk",
        "dv",
        "chunk_size",
        "data_type",
        "state_data_type",
        "has_g",
        "is_continue",
        "query_datarange",
        "key_datarange",
        "value_datarange",
        "gamma_datarange",
        "beta_datarange",
        "state_datarange",
        "errmsg",
        "durations",
    ]

    append = os.environ.get("CSV_APPEND", "0") == "1"
    mode = "a" if append else "w"
    write_header = not (
        append and os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    )

    with open(csv_file, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(_RESULT_ROWS)

    total = len(_RESULT_ROWS)
    passed = sum(1 for r in _RESULT_ROWS if r["status"] == "PASSED")
    failed = sum(1 for r in _RESULT_ROWS if r["status"] in ("FAILED", "ERROR"))
    logger.info(
        f"\nCSV generated: {csv_file} (total {total}, passed{passed}, failed{failed})"
    )
