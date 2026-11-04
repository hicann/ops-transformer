# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DSL publication and optional dependency integration with the ASC build."""

import ctypes
import importlib.util
import json
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / "cmake/scripts/build_cannbotdsl.py"


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def cmake(*arguments, succeeds=True):
    result = subprocess.run(
        ["cmake", *map(str, arguments)], capture_output=True, text=True
    )
    assert (result.returncode == 0) == succeeds, result.stdout + result.stderr
    return result


def build_install(project):
    cmake("--build", project / "build")
    cmake("--install", project / "build", "--prefix", project / "installed")
    (library,) = (project / "installed").rglob("*.so")
    loaded = ctypes.CDLL(str(library))
    assert loaded.asc_entry() == 17
    return loaded


@pytest.fixture
def builder(monkeypatch):
    spec = importlib.util.spec_from_file_location("dsl_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "framework_files", lambda: [])
    monkeypatch.setattr(
        module, "source_exports", lambda source: dict(native=True, aclnn=False)
    )
    return module


def test_publication_cache_tracks_inputs_and_preserves_previous_package(
    builder, tmp_path, monkeypatch
):
    source = write(tmp_path / "collector/entry.py", "old")
    compiler = write(tmp_path / "compiler.so", b"compiler-v1")
    monkeypatch.setattr(builder, "framework_files", lambda: [compiler])
    calls, fail = [], False

    def publish(**options):
        calls.append(options["target"])
        write(options["output"] / "archive.a", source.read_text())
        if fail:
            raise RuntimeError("compiler failed")

    monkeypatch.setitem(sys.modules, "opkit", types.SimpleNamespace(build=publish))
    output = tmp_path / "host"
    builder.build(source, output, "ascend950", "host")
    before = (output / "archive.a").stat().st_mtime_ns
    builder.build(source, output, "ascend950", "host")
    assert len(calls) == 1 and (output / "archive.a").stat().st_mtime_ns == before
    compiler.write_bytes(b"compiler-v2")
    builder.build(source, output, "ascend950", "host")
    assert len(calls) == 2
    source.write_text("new")
    fail = True
    with pytest.raises(RuntimeError, match="compiler failed"):
        builder.build(source, output, "ascend950", "host")
    assert (output / "archive.a").read_text() == "old"
    fail = False
    builder.build(source, output, "ascend950", "host")
    assert (output / "archive.a").read_text() == "new"
    (output / "archive.a").write_text("damaged")
    builder.build(source, output, "ascend950", "host")
    assert (output / "archive.a").read_text() == "new" and len(calls) == 5
    builder.build(source, output, "ascend960dt", "host", target="configured-target")
    state = json.loads((output / ".transformer-build.json").read_text())
    assert state["record"]["inputs"]["target"] == calls[-1] == "configured-target"
    assert state["record"]["inputs"]["soc"] == "ascend960dt" and len(calls) == 6


def test_atomic_replacement_restores_previous_package(builder, tmp_path, monkeypatch):
    output = tmp_path / "package"
    write(output / "valid", "previous")
    rename = Path.rename

    def fail_publish(path, destination):
        if path.name == "package" and path != output:
            raise OSError("replace failed")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="replace failed"):
        with builder.publication(output) as stage:
            write(stage / "new", "new")
    assert (output / "valid").read_text() == "previous"


def test_input_dependencies_preserve_cmake_paths(builder, tmp_path, monkeypatch):
    source = write(tmp_path / "entry.py", "# collector")
    dependencies = [
        write(tmp_path / name, "# dependency")
        for name in (
            "space path.py",
            "semicolon;path.py",
            "中文输入.py",
        )
    ]
    monkeypatch.setattr(builder, "framework_files", lambda: dependencies)
    inputs = tmp_path / "inputs.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER),
            "--mode",
            "inputs",
            "--source",
            str(source),
            "--output",
            str(inputs),
        ],
    )
    builder.main()
    script = write(
        tmp_path / "dependencies.cmake",
        """
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/inputs.list" dependencies ENCODING UTF-8)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${dependencies}")
get_property(dependencies DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
foreach(path IN LISTS dependencies)
    file(APPEND "${CMAKE_CURRENT_LIST_DIR}/actual.list" "${path}\\n")
endforeach()
""",
    )
    cmake("-P", script)
    assert set((tmp_path / "actual.list").read_text().splitlines()) == set(
        json.loads(inputs.read_text())["files"]
    )


def test_kernel_scan_replacement_respects_ownership(builder, tmp_path):
    package = tmp_path / "first"
    original = write(package / "kernel/ascend950/sample/Old.o", b"DSL")
    bins = tmp_path / "binary/ascend950/bin"
    neighbor = write(bins / "sample/asc.o", b"ASC")
    builder.sync_kernels(package, bins, "ascend950")
    original.unlink()
    write(package / "kernel/ascend950/sample/New.o", b"new DSL")
    builder.sync_kernels(package, bins, "ascend950")
    assert not (bins / "sample/Old.o").exists()
    assert (bins / "sample/New.o").read_bytes() == b"new DSL"
    assert neighbor.read_bytes() == b"ASC"
    other = tmp_path / "second"
    write(other / "kernel/ascend950/sample/New.o", b"other")
    with pytest.raises(ValueError, match="conflicting"):
        builder.sync_kernels(other, bins, "ascend950")
    foreign = write(tmp_path / "foreign/sample/New.o", b"ASC")
    with pytest.raises(ValueError, match="unowned"):
        builder.sync_kernels(package, foreign.parent.parent, "ascend950")
    assert foreign.read_bytes() == b"ASC"


def _optional_dsl_project(tmp_path, scenario):
    dependencies = tmp_path / "dependencies"
    files = {
        "opkit/__init__.py": (
            "def build(**kwargs):\n    raise RuntimeError('DSL compiler failure')\n"
        ),
        "cannbotdsl/__init__.py": "",
        "cannbotdsl/aot/__init__.py": "",
        "cannbotdsl/aot/collection.py": (
            "def _declares_registration(source, markers): return 'export' in markers\n"
        ),
        "cannbotdsl/core/__init__.py": "",
        "cannbotdsl/core/toolchain/__init__.py": "",
        "cannbotdsl/core/toolchain/ascendc.py": (
            "from types import SimpleNamespace\n"
            f"def resolve_bisheng(): return SimpleNamespace(path={sys.executable!r})\n"
        ),
    }
    if scenario == "bad_import":
        files["cannbotdsl/__init__.py"] = "raise OSError('broken DSL import')\n"
    for relative, content in files.items():
        write(dependencies / relative, content)
    sdk = tmp_path / "sdk"
    bootstrap = tmp_path / "probe.py"
    calls = tmp_path / "probe-calls.log"
    blocked = {"dsl": "cannbotdsl", "dsl_stale": "cannbotdsl", "opkit": "opkit"}.get(
        scenario
    )
    bootstrap.write_text(f"""
import importlib.abc, pathlib, runpy, sys
sys.path.insert(0, {str(dependencies)!r})
sys.argv = sys.argv[1:]
mode = 'check' if sys.argv[0] == '-c' else sys.argv[sys.argv.index('--mode') + 1]
with pathlib.Path({str(calls)!r}).open('a') as stream: stream.write(mode + '\\n')
class MissingDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {blocked!r}:
            with pathlib.Path({str(calls.with_suffix(".lookups"))!r}).open('a') as stream:
                stream.write(mode + ':' + fullname + '\\n')
            raise ModuleNotFoundError('blocked optional dependency: ' + fullname, name=fullname)
sys.meta_path.insert(0, MissingDependency())
if mode == 'check':
    exec(sys.argv[1], {{'__name__': '__main__'}})
else:
    runpy.run_path(sys.argv[0], run_name='__main__')
""")
    launcher = tmp_path / "probe-python"
    launcher.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " -I "
        + shlex.quote(str(bootstrap))
        + ' "$@"\n'
    )
    launcher.chmod(0o755)
    for name in ("first", "second"):
        write(
            tmp_path / name / (name + ".py"),
            "raise AssertionError('unexpected collection')\n",
        )
    (tmp_path / "asc.cpp").write_text('extern "C" int asc_entry() { return 17; }')
    (tmp_path / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.18.4)
project(compatibility CXX)
include("{ROOT}/cmake/func.cmake")
set(OPS_ADV_DIR "{ROOT}")
set(HI_PYTHON "{launcher}")
set(ASCEND_CANN_PACKAGE_PATH "{sdk}")
set(ASCEND_BINARY_OUT_DIR "${{CMAKE_BINARY_DIR}}/binary")
set(ASCEND_COMPUTE_UNIT ascend950)
set(CANNBOTDSL_COMPUTE_UNIT ascend950)
set(CANNBOTDSL_TARGET dav-3510)
set(ENABLE_OPS_HOST ON)
set(ENABLE_OPS_KERNEL ON)
add_library(asc SHARED asc.cpp)
set_target_properties(asc PROPERTIES SUFFIX ".so")
install(TARGETS asc LIBRARY DESTINATION lib)
enable_cannbotdsl_kernel(first "{tmp_path}/first")
enable_cannbotdsl_kernel(second "{tmp_path}/second")
cannbotdsl_clean_resources()
''')
    return calls


@pytest.mark.parametrize("scenario", ["dsl", "opkit"])
def test_real_unavailable_probe_still_builds_and_installs_asc(tmp_path, scenario):
    calls = _optional_dsl_project(tmp_path, scenario)
    build = tmp_path / "build"
    result = cmake("-S", tmp_path, "-B", build)
    for name in ("first", "second"):
        assert f"DSL compilation skipped (not compiled): {name}" in result.stderr
    assert "cannbotdsl or opkit is unavailable in" in " ".join(result.stderr.split())
    assert "Traceback" not in result.stderr
    assert calls.read_text() == "check\n"
    build_install(tmp_path)
    assert not (build / "cannbot_dsl").exists()
    assert calls.read_text() == "check\n"


def test_real_unavailable_probe_cleans_owned_stale_resources_and_keeps_asc(tmp_path):
    calls = _optional_dsl_project(tmp_path, "dsl_stale")
    build = tmp_path / "build"
    root = build / "cannbot_dsl"
    package = root / "first/device"
    resources = package / "kernel/ascend950/retired"
    resources.mkdir(parents=True)
    bins = build / "binary/ascend950/bin"
    retired = bins / "retired"
    retired.mkdir(parents=True)
    owned = ["retired/Old.o", "retired/Old.json"]
    for name in ("Old.o", "Old.json"):
        (resources / name).write_bytes(b"old DSL")
        (retired / name).write_bytes(b"old DSL")
    (bins.parent / ".cannbotdsl-files.json").write_text(
        json.dumps({str(package): owned})
    )
    (bins / "retired.json").write_text("old DSL summary")
    index_names = ("binary_info_config.json", "relocatable_kernel_info_config.json")
    for name in index_names:
        (bins / name).write_text("old DSL index")
    asc_neighbor = bins / "asc.o"
    asc_neighbor.write_bytes(b"ASC neighbor")
    stamp = root / "first/kernel.built"
    stamp.touch()
    metadata = root / "metadata/host.json"
    metadata.parent.mkdir()
    metadata.write_text("old DSL metadata")
    registry = build / "opkit-native/host/registry.cpp"
    registry.parent.mkdir(parents=True)
    registry.write_text("old DSL registry")

    cmake("-S", tmp_path, "-B", build)
    build_install(tmp_path)
    assert calls.read_text() == "check\nclean\n"
    assert calls.with_suffix(".lookups").read_text() == "check:cannbotdsl\n"
    assert not retired.exists() and not (bins / "retired.json").exists()
    assert not any((bins / name).exists() for name in index_names)
    assert not stamp.exists() and not metadata.exists() and not registry.exists()
    assert asc_neighbor.read_bytes() == b"ASC neighbor"
    assert json.loads((bins.parent / ".cannbotdsl-files.json").read_text()) == {}
    assert not list((tmp_path / "installed").rglob("Old.*"))


@pytest.mark.parametrize(
    "scenario,reason,modes",
    [
        ("bad_import", "broken DSL import", ["check", "inputs"]),
        ("build_failure", "DSL compiler failure", ["check", "inputs", "host"]),
    ],
)
def test_discoverable_modules_do_not_hide_real_dsl_build_errors(
    tmp_path, scenario, reason, modes
):
    calls = _optional_dsl_project(tmp_path, scenario)
    (tmp_path / "first/first.py").write_text("# collector\n")
    result = cmake("-S", tmp_path, "-B", tmp_path / "build", succeeds=False)
    assert reason in result.stdout + result.stderr
    assert "DSL compilation skipped" not in result.stderr
    assert calls.read_text().splitlines() == modes
