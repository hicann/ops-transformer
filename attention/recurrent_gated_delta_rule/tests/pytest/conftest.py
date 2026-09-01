# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of the
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE. See the License for the specific language governing permissions and limitations under the
# License.
# -----------------------------------------------------------------------------------------------------------

import csv
import logging
import os

import pytest
import torch  # noqa: F401
import torch_npu  # noqa: F401

logger = logging.getLogger(__name__)

# 与 test_recurrent_gated_delta_rule_single.py 的 param_names 一一对应的全部入参列（除随机种子外均取自用例参数）
PARAM_COLUMNS = [
    "batch_size",
    "mtp",
    "nk",
    "nv",
    "dk",
    "dv",
    "actual_seq_lengths",
    "ssm_state_indices",
    "has_gamma",
    "has_gamma_k",
    "has_num_accepted_tokens",
    "scale_value",
    "num_accepted_tokens",
    "block_num",
    "data_type",
    "state_data_type",
    "query_datarange",
    "key_datarange",
    "value_datarange",
    "gamma_datarange",
    "gamma_k_datarange",
    "beta_datarange",
    "state_datarange",
    "state_non_contiguous",
]

_FIELDS = (
    [
        "random_seed",
        "tensor_seed",
        "test_mode",
        "check_type",
        "result",
        "mss_check",
        "out_pct_rlt",
        "state_pct_rlt",
    ]
    + PARAM_COLUMNS
    + ["errmsg"]
)

_RESULT_ROWS = []


def _get_check_type():
    """precision=带CPU golden精度对比；
    execution_only=仅NPU执行（random_npu模式）；
    execution_only+mss_<tool>=mssanitizer检测（mss模式，tool为memcheck/racecheck/initcheck/synccheck）。"""
    if os.environ.get("SKIP_GOLDEN", "0") != "1":
        return "precision"
    mss_tool = os.environ.get("MSS_TOOL", "")
    if mss_tool:
        return f"execution_only+mss_{mss_tool}"
    if os.environ.get("MSS_CHECK", "0") == "1":
        return "execution_only+mss_memcheck"
    return "execution_only"


def _get_precision_pcts():
    """读取 golden 层记录的本条用例精度达标率（PctRlt）。
    仅 precision 模式有值；execution_only（SKIP_GOLDEN）无精度对比，保持空。"""
    try:
        import recurrent_gated_delta_rule_golden as golden

        return golden.LAST_OUT_PCT, golden.LAST_STATE_PCT
    except Exception:
        return None, None


def _fmt_pct(value):
    if value is None:
        return ""
    return f"{value:.6f}%"


def _format_param(value):
    """list/张量类型转可读字符串，None 保留为空。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


_CURRENT_TENSOR_SEED = 0


@pytest.fixture(autouse=True)
def _set_tensor_seed():
    """每条用例前固定 torch 随机数种子（张量数值可复现）。
    TORCH_SEED 环境变量指定固定种子；否则自动生成并回写，供 conftest 记 CSV。"""
    global _CURRENT_TENSOR_SEED
    fix_seed = os.environ.get("TORCH_SEED", "")
    if fix_seed:
        _CURRENT_TENSOR_SEED = int(fix_seed)
    else:
        _CURRENT_TENSOR_SEED = int(torch.seed() % (2**31))
    torch.manual_seed(_CURRENT_TENSOR_SEED)
    torch.npu.manual_seed(_CURRENT_TENSOR_SEED)
    yield


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

    out_pct, state_pct = _get_precision_pcts()
    row = {
        "random_seed": os.environ.get("RANDOM_SEED", ""),
        "tensor_seed": str(_CURRENT_TENSOR_SEED),
        "test_mode": os.environ.get("TEST_MODE", ""),
        "check_type": _get_check_type(),
        "result": status,
        "mss_check": "",
        "out_pct_rlt": _fmt_pct(out_pct),
        "state_pct_rlt": _fmt_pct(state_pct),
    }
    for col in PARAM_COLUMNS:
        row[col] = _format_param(params.get(col, ""))
    row["errmsg"] = error
    _RESULT_ROWS.append(row)


def pytest_sessionfinish(session, exitstatus):
    csv_file = os.environ.get("CSV_FILE", "")
    if not csv_file or not _RESULT_ROWS:
        return

    append = os.environ.get("CSV_APPEND", "0") == "1"
    mode = "a" if append else "w"
    write_header = not (
        append and os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    )

    with open(csv_file, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(_RESULT_ROWS)

    total = len(_RESULT_ROWS)
    passed = sum(1 for r in _RESULT_ROWS if r["result"] == "PASSED")
    failed = sum(1 for r in _RESULT_ROWS if r["result"] in ("FAILED", "ERROR"))
    logger.info(
        f"CSV generated: {csv_file} (total {total}, passed {passed}, failed {failed})"
    )
