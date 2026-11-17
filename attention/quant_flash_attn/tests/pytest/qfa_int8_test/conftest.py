# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP8-compatible cache stages and profiling options for INT8."""

import os
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

_STAGES = {"gen", "cpu", "npu", "compare"}


def pytest_addoption(parser):
    group = parser.getgroup("QFA INT8")
    group.addoption(
        "--golden-mode",
        default="all",
        help="all or a comma-separated gen,cpu,npu,compare subset",
    )
    group.addoption("--cache-dir", default=None, help="Golden .pt cache directory")
    group.addoption(
        "--msprof",
        action="store_true",
        help="Run pytest under msprof and parse its report",
    )
    group.addoption(
        "--parse-prof", default=None, help="Parse an existing PROF directory"
    )
    group.addoption(
        "--perf-baseline",
        default=None,
        help="Compare against a saved performance report",
    )
    group.addoption(
        "--perf-threshold",
        type=float,
        default=8.0,
        help="Allowed performance regression percentage",
    )


def pytest_configure(config):
    for marker in (
        "func_rdv: public API accuracy cases",
        "perf_rdv: profiled model cases",
        "ci: representative functional cases",
        "contract: API, graph and TestSpec checks",
    ):
        config.addinivalue_line("markers", marker)
    modes = {part.strip() for part in config.getoption("--golden-mode").split(",")}
    if not modes or modes - (_STAGES | {"all"}):
        raise pytest.UsageError("--golden-mode expects all or gen,cpu,npu,compare")
    if not (config.getoption("--msprof") or config.getoption("--parse-prof")):
        return
    # Reuse the FP8 report parser instead of maintaining an INT8 copy.
    # Loading common/__init__.py would import the FP8 runtime and its JIT ops.
    path = Path(__file__).resolve().parent.parent / "qfa_fp8_test/common/perf_parser.py"
    spec = importlib.util.spec_from_file_location("qfa_shared_perf_parser", path)
    perf_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_parser)

    status = 0
    directory = config.getoption("--parse-prof")
    if config.getoption("--msprof"):
        previous = set(Path.cwd().glob("PROF_*"))
        status = subprocess.call(
            [
                "msprof",
                sys.executable,
                "-m",
                "pytest",
                *[arg for arg in sys.argv[1:] if arg != "--msprof"],
            ]
        )
        generated = set(Path.cwd().glob("PROF_*")) - previous
        if not generated:
            pytest.exit("msprof produced no PROF directory", returncode=status or 1)
        directory = str(max(generated, key=lambda path: path.stat().st_mtime))
    results, csv_path = perf_parser.parse_prof_directory(
        directory, operator_name="QuantFlashAttn"
    )
    if not results:
        pytest.exit("no QuantFlashAttn entries found in profiling output", returncode=1)
    perf_parser.print_report(results)
    perf_parser.save_report(results, csv_path, os.path.join(os.getcwd(), "perf_output"))
    baseline = config.getoption("--perf-baseline")
    if baseline:
        comparisons = perf_parser.compare_with_baseline(
            results,
            perf_parser.parse_baseline_log(baseline),
            config.getoption("--perf-threshold"),
        )
        perf_parser.print_comparison_report(
            comparisons, config.getoption("--perf-threshold")
        )
        if any(row["status"] == "FAILED" for row in comparisons):
            status = status or 1
    pytest.exit("profiling report complete", returncode=status)


@pytest.fixture(scope="session")
def int8_npu():
    npu = pytest.importorskip("torch_npu").npu
    if not npu.is_available():
        pytest.skip("QFA INT8 requires an Atlas A2/A3 NPU")

    device = int(os.environ.get("QFA_TEST_NPU_DEVICE", "0"))
    name = npu.get_device_name(device)
    if not name.startswith(("Ascend910B", "Ascend910_93")):
        pytest.skip(f"QFA INT8 requires an Atlas A2/A3 NPU; found {name}")


@pytest.fixture(scope="session")
def golden_mode(request):
    modes = {
        part.strip() for part in request.config.getoption("--golden-mode").split(",")
    }
    if "all" in modes:
        modes = _STAGES
    if "npu" in modes:
        request.getfixturevalue("int8_npu")
    return modes


@pytest.fixture(scope="session")
def cache_dir(request):
    return request.config.getoption("--cache-dir")
