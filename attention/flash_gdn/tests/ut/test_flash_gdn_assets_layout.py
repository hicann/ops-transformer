# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression checks for the standard assets/spec.py + impl/cases layout."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

TESTS = Path(__file__).resolve().parents[1]
IMPL = TESTS / "assets/impl"
sys.path.insert(0, str(IMPL))
import gdn_ttk_ops


@pytest.mark.parametrize("name", ["golden", "inputs", "compare", "peer"])
def test_impl_modules_resolve_from_assets_impl(name):
    module = gdn_ttk_ops.load_asset(name)
    assert Path(module.__file__).resolve() == (IMPL / (name + ".py")).resolve()
    assert not hasattr(module, "__spec__") or not isinstance(module.__spec__, dict)


def test_all_ttk_resources_live_under_assets():
    assets = TESTS / "assets"
    assert not (TESTS / "ttk").exists()
    assert not (TESTS / "gdn_ttk_ops.py").exists()
    assert (assets / "spec.py").is_file()
    assert (assets / "cpu_reference.yaml").is_file()
    assert {path.name for path in (assets / "cases").glob("*.csv")} == {
        "flash_gdn_smoke.csv",
        "flash_gdn_a5_redline.csv",
    }


def test_standalone_spec_loads_csv_api_lazily_without_test_pythonpath(tmp_path):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in env.get("PYTHONPATH", "").split(os.pathsep)
        if value and Path(value).resolve() not in (TESTS.resolve(), IMPL.resolve())
    )
    code = r"""
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = importlib.util.spec_from_file_location("gdn_standalone_spec_probe", path)
module = importlib.util.module_from_spec(source)
sys.modules[source.name] = module
source.loader.exec_module(module)
assert set(module.__spec__) == {"gdn_ttk_ops.flash_gdn", "gdn_ttk_ops.flash_gdn_fwd"}
for name in ("golden", "inputs", "peer", "compare"):
    assert "_flash_gdn_test_" + name not in sys.modules
assert "torch_npu" not in sys.modules
adapter = importlib.import_module("gdn_ttk_ops")
assert Path(adapter.__file__).resolve() == (path.parent / "impl/gdn_ttk_ops.py").resolve()
for api, cls in module.__spec__.items():
    function = getattr(adapter, api.rsplit(".", 1)[1])
    assert "metadata" in inspect.signature(function).parameters
    spec = getattr(module, cls)
    assert spec.tolerance == {dtype: {"standard": "cross_check", "level": "L2"}
                              for dtype in ("bfloat16", "float32")}
    assert callable(spec.golden) and callable(spec.customize_inputs)
print("STANDALONE_GDN_SPEC_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(TESTS / "assets/spec.py")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STANDALONE_GDN_SPEC_OK" in result.stdout
