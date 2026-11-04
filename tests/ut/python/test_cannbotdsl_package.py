# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DSL source wheel packaging and custom Torch package imports."""

import builtins
import importlib.util
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "torch_extension/cann_ops_transformer"


@pytest.mark.parametrize("vendor", [None, "review"])
def test_wheel_includes_recursive_dsl_sources_with_working_relative_imports(
    tmp_path, vendor
):
    project = tmp_path / "project"
    setup_dir = project / "torch_extension"
    base = setup_dir / "cann_ops_transformer"
    (base / "ops/attention").mkdir(parents=True)
    for directory in (base, base / "ops", base / "ops/attention"):
        (directory / "__init__.py").write_text("")
    shutil.copy2(ROOT / "torch_extension/setup.py", setup_dir / "setup.py")
    expected_sources = {}
    for name in ("sample", "other"):
        op = project / "attention" / name
        (op / "torch_extension").mkdir(parents=True)
        (op / "torch_extension/__init__.py").write_text(
            "def sample():\n    from .op_dsl.sample import compile_sample\n    return compile_sample()\n"
        )
        (op / "op_dsl/helpers").mkdir(parents=True)
        for relative, content in {
            "sample.py": "from .helpers.constants import VALUE\ndef compile_sample():\n    return VALUE\n",
            "helpers/__init__.py": "",
            "helpers/constants.py": "VALUE = 17\n",
        }.items():
            (op / "op_dsl" / relative).write_text(content)
            if name == "sample":
                expected_sources[relative] = content
        (op / "op_dsl/README.md").write_text("build documentation")
        (op / "op_dsl/temporary.o").write_bytes(b"not a runtime source")
        (op / "op_dsl/__pycache__").mkdir()
        (op / "op_dsl/__pycache__/stale.py").write_text("raise RuntimeError('stale')")
    environment = os.environ.copy()
    for key in (
        "TORCH_EXTENSION_OPS",
        "TORCH_EXTENSION_VENDOR",
        "TORCH_EXTENSION_EXPERIMENTAL",
    ):
        environment.pop(key, None)
    if vendor:
        environment.update(TORCH_EXTENSION_OPS="sample", TORCH_EXTENSION_VENDOR=vendor)
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=setup_dir,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = (setup_dir / "dist").glob("*.whl")
    package = "cann_ops_transformer" + ("_" + vendor if vendor else "")
    prefix = package + "/ops/attention/sample/op_dsl/"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert {
            name[len(prefix) :] for name in names if name.startswith(prefix)
        } == set(expected_sources)
        for relative, content in expected_sources.items():
            assert archive.read(prefix + relative).decode() == content
        assert any("/other/op_dsl/" in name for name in names) == (vendor is None)
        installed = tmp_path / "installed"
        archive.extractall(installed)
    for key in ("ASCEND_OPP_PATH", "ASCEND_HOME_PATH", "ASCEND_CUSTOM_OPP_PATH"):
        environment.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from "
            + package
            + ".ops.attention.sample import sample; assert sample() == 17",
            str(installed),
        ],
        text=True,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_custom_builder_loads_without_base_package(tmp_path, monkeypatch):
    namespace = "cann_ops_transformer_custom"
    pytest.importorskip("torch")
    for suffix, directory in [
        ("", PACKAGE),
        (".op_builder", PACKAGE / "op_builder"),
        (".utils", PACKAGE / "utils"),
    ]:
        package = types.ModuleType(namespace + suffix)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, namespace + suffix, package)
    name = namespace + ".op_builder.builder"
    spec = importlib.util.spec_from_file_location(
        name, PACKAGE / "op_builder/builder.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    class Builder(module.OpBuilder):
        def sources(self):
            return []

        def schema(self):
            return None

        def register_meta(self):
            pass

    builder = Builder("sample")
    monkeypatch.setattr(builder, "_ensure_initialized", lambda: None)
    monkeypatch.setattr(builder, "_remove_stale_torch_lock", lambda: None)
    extension = types.SimpleNamespace(add=lambda x, y: x + y)
    monkeypatch.setattr(builder, "_try_load_cached", lambda wrap, path: wrap(extension))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    original_import = builtins.__import__

    def without_base_package(name, *args, **kwargs):
        if name == "cann_ops_transformer" or name.startswith("cann_ops_transformer."):
            raise ModuleNotFoundError("Base package is not installed", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_base_package)
    assert builder.load().add(2, 3) == 5
