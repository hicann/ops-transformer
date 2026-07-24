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
import subprocess
import sys
from pathlib import Path

import pytest

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
        help=(
            "Golden 执行模式，支持逗号组合: "
            "all=全流程, gen=生成数据, cpu=跑CPU, npu=跑NPU, compare=精度对比. "
            "例: --golden-mode=npu,compare"
        ),
    )
    parser.addoption(
        "--cache-dir",
        default=None,
        help="golden 缓存目录路径（默认 common/golden_cache/）",
    )
    parser.addoption(
        "--parse-prof",
        default=None,
        metavar="PROF_DIR",
        help=(
            "解析 msprof PROF 目录中的 op_summary.csv 并输出性能报告。"
            "例: --parse-prof=./PROF_000001_xxx"
        ),
    )
    parser.addoption(
        "--msprof",
        action="store_true",
        default=False,
        help=(
            "自动用 msprof 包裹 pytest 运行，测试完成后自动解析 PROF 目录并输出性能报告。"
            "例: pytest --msprof -v -m debug"
        ),
    )
    parser.addoption(
        "--perf-baseline",
        default=None,
        metavar="BASELINE_LOG",
        help=(
            "性能基线 log 文件路径，与当前结果比较 Duration。"
            "劣化超过阈值（默认 8%%）标记为 FAILED。"
            "例: --perf-baseline=./perf_baseline/perf_report_xxx.log"
        ),
    )
    parser.addoption(
        "--perf-threshold",
        default=8.0,
        type=float,
        metavar="PERCENT",
        help="性能劣化阈值（百分比），超过此值标记为 FAILED（默认 8.0）",
    )


def _compare_baseline(config, results, tag="perf"):
    baseline_path = config.getoption("--perf-baseline", default=None)
    if not baseline_path:
        return False

    from common import perf_parser

    threshold = config.getoption("--perf-threshold", default=8.0)
    baseline = perf_parser.parse_baseline_log(baseline_path)
    if not baseline:
        print(f"[{tag}] Warning: baseline file not found or empty: {baseline_path}")
        return False

    print(f"\n[{tag}] Comparing with baseline: {baseline_path}")
    comparison = perf_parser.compare_with_baseline(results, baseline, threshold)
    perf_parser.print_comparison_report(comparison, threshold)
    failed = sum(1 for r in comparison if r["status"] == "FAILED")
    if failed > 0:
        pytest.exit("performance regression detected", returncode=1)
    return True


def _run_msprof_and_parse(config):
    from common import perf_parser

    cwd = os.getcwd()
    existing = {p.name for p in Path(cwd).glob("PROF_*") if p.is_dir()}

    inner_args = [a for a in sys.argv[1:] if a != "--msprof"]
    cmd = ["msprof", sys.executable, "-m", "pytest"] + inner_args
    print(f"[msprof] Running: {' '.join(cmd)}")
    print(f"[msprof] Working directory: {cwd}")
    print()

    ret = subprocess.call(cmd)

    print()
    all_dirs = sorted(
        (p for p in Path(cwd).glob("PROF_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    new_dirs = [str(p) for p in all_dirs if p.name not in existing]

    if not new_dirs:
        print(f"[msprof] No new PROF directory found (exit code: {ret})")
        print(f"[msprof] Existing PROF dirs: {sorted(existing)}")
        pytest.exit("msprof run complete but no PROF output found", returncode=ret)

    prof_dir = new_dirs[0]
    print(f"[msprof] Tests finished (exit code: {ret})")
    print(f"[msprof] Parsing: {prof_dir}")

    results, csv_path = perf_parser.parse_prof_directory(prof_dir)
    perf_parser.print_report(results)
    log_file = perf_parser.save_report(results, csv_path, os.path.join(cwd, "perf_output"))
    print(f"[msprof] Report saved: {log_file}")

    _compare_baseline(config, results, tag="msprof")
    pytest.exit("msprof profiling complete", returncode=ret)


def pytest_configure(config):
    if config.getoption("--msprof", default=False):
        _run_msprof_and_parse(config)
        return

    prof_dir = config.getoption("--parse-prof", default=None)
    if not prof_dir:
        return

    from common import perf_parser

    results, csv_path = perf_parser.parse_prof_directory(prof_dir)
    perf_parser.print_report(results)
    log_file = perf_parser.save_report(results, csv_path, os.path.join(os.getcwd(), "perf_output"))
    print(f"[perf] Report saved: {log_file}")

    _compare_baseline(config, results, tag="perf")
    pytest.exit("profiling report complete", returncode=0)


@pytest.fixture(scope="session")
def golden_mode(request):
    return _parse_golden_mode(request.config.getoption("--golden-mode"))


@pytest.fixture(scope="session")
def cache_dir(request):
    return request.config.getoption("--cache-dir")
