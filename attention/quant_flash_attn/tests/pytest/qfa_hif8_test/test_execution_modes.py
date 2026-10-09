#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

D = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = load("qfa_config_under_test", D / "conftest.py")


@pytest.mark.parametrize(
    "raw,is_perf,check,expected",
    [
        (None, True, False, {"perf"}),
        (None, True, True, {"gen", "cpu", "npu", "compare"}),
        (None, False, False, {"gen", "cpu", "npu", "compare"}),
        ("npu,compare", True, False, {"npu", "compare"}),
        ("perf", False, False, {"perf"}),
    ],
)
def test_mode_resolution(raw, is_perf, check, expected):
    assert config._resolve_golden_mode(raw, is_perf, check) == expected


@pytest.mark.parametrize("raw", ["", "perf,npu", "perf,all", "typo"])
def test_invalid_mode(raw):
    with pytest.raises(pytest.UsageError):
        config._parse_golden_mode(raw)


def test_conflicting_flags():
    options = {"--check-accuracy": True, "--golden-mode": "perf"}
    with pytest.raises(pytest.UsageError):
        config.pytest_configure(types.SimpleNamespace(getoption=options.get))


@pytest.fixture
def setup_runner(monkeypatch):
    cache = Mock()
    golden = Mock()
    compare = Mock()
    torch = types.ModuleType("torch")
    torch.profiler = types.SimpleNamespace(
        record_function=lambda _: contextlib.nullcontext()
    )
    torch.npu = Mock()
    common = types.ModuleType("common")
    common.golden_cache = cache
    common.quant_flash_attn_golden = golden
    common.result_compare_method = compare
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "common", common)
    runner = load("qfa_runner_under_test", D / "common/test_runner.py")
    data = tuple(object() for _ in range(10))
    cache.load_input.return_value = data
    golden.generate_data.return_value = data
    golden.npu_hif8_fa.return_value = (object(), object())
    golden.cpu_hif8_golden.return_value = (object(), object())
    golden.INPUT_LAYOUT = "BNSD"
    golden.ENABLE_LSE = False
    compare.check_result.return_value = "Pass"
    return runner, cache, golden, compare, torch


@pytest.mark.parametrize("cache_hit", [True, False])
def test_perf_skips_golden_and_output_copy(setup_runner, cache_hit):
    runner, cache, golden, compare, torch = setup_runner
    cache.has_input.return_value = cache_hit
    assert runner.execute_test({"name": "target"}, {"perf"}, "/cache") == (None, None)
    assert golden.generate_data.call_count == int(not cache_hit)
    assert cache.save_input.call_count == int(not cache_hit)
    assert cache.load_input.call_count == int(cache_hit)
    golden.npu_hif8_fa.assert_called_once()
    torch.npu.synchronize.assert_called_once()
    golden.cpu_hif8_golden.assert_not_called()
    cache.load_cpu_output.assert_not_called()
    cache.save_cpu_output.assert_not_called()
    cache.save_npu_output.assert_not_called()
    compare.check_result.assert_not_called()


@pytest.mark.parametrize("error_source", ["npu", "sync"])
def test_perf_device_errors_fail(setup_runner, error_source):
    runner, cache, golden, compare, torch = setup_runner
    call = golden.npu_hif8_fa if error_source == "npu" else torch.npu.synchronize
    call.side_effect = RuntimeError("device failure")
    with pytest.raises(RuntimeError, match="device failure"):
        runner.execute_test({"name": "target"}, {"perf"})


def test_full_mode_still_checks_accuracy(setup_runner):
    runner, cache, golden, compare, torch = setup_runner
    result = runner.execute_test({"name": "target"}, {"gen", "cpu", "npu", "compare"})
    golden.cpu_hif8_golden.assert_called_once()
    cache.save_cpu_output.assert_called_once()
    cache.save_npu_output.assert_called_once()
    compare.check_result.assert_called_once()
    assert result == ("Pass", None)
    with pytest.raises(pytest.fail.Exception):
        runner.check_results("Fail", None)


def test_legacy_npu_only_does_not_require_golden(setup_runner):
    runner, cache, golden, compare, torch = setup_runner
    assert runner.execute_test({"name": "target"}, {"npu"}) == (None, None)
    golden.cpu_hif8_golden.assert_not_called()
    cache.load_cpu_output.assert_not_called()
    cache.save_npu_output.assert_called_once()


@pytest.mark.parametrize(
    "args,expect_perf,expect_error",
    [
        ([], True, False),
        (["--check-accuracy"], False, False),
        (["--golden-mode=perf"], True, False),
        (["--golden-mode=npu,compare"], False, False),
        (["--golden-mode=perf,npu"], False, True),
        (["--check-accuracy", "--golden-mode=perf"], False, True),
    ],
)
def test_real_pytest_cli(tmp_path, args, expect_perf, expect_error):
    import os
    import shutil
    import subprocess

    shutil.copyfile(D / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    perf_rdv: performance case\n"
    )
    expected = (
        {"perf"}
        if expect_perf
        else (
            {"npu", "compare"}
            if "--golden-mode=npu,compare" in args
            else {"gen", "cpu", "npu", "compare"}
        )
    )
    functional = (
        {"perf"}
        if "--golden-mode=perf" in args
        else (
            {"npu", "compare"}
            if "--golden-mode=npu,compare" in args
            else {"gen", "cpu", "npu", "compare"}
        )
    )
    (tmp_path / "test_modes.py").write_text(
        "import pytest\n@pytest.mark.perf_rdv\ndef test_perf(golden_mode):\n"
        f"    assert golden_mode == {expected!r}\n"
        f"def test_functional(golden_mode):\n    assert golden_mode == {functional!r}\n"
    )
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--junitxml=results.xml"] + args,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    if expect_error:
        assert result.returncode == 4, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert ("without accuracy comparison" in result.stdout) == expect_perf
        if expect_perf:
            assert "accuracy_checked" in (tmp_path / "results.xml").read_text()
            assert "False" in (tmp_path / "results.xml").read_text()
