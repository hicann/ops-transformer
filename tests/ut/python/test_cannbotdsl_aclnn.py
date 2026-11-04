# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLNN source publication and linkage alongside the existing Native path."""

import ctypes
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from test_cannbotdsl_build import ROOT, BUILDER, builder, cmake, write


def publish_aclnn(root, *, mode, l0, l2, **unused):
    if mode == "kernel":
        write(root / "kernel/ascend950/sample/Sample.o", b"device")
        write(root / "kernel/ascend950/sample/Sample.json", "{}")
        return
    write(root / "runtime/runtime.cpp", 'extern "C" int runtime_value() { return 17; }')
    write(
        root / "runtime/CMakeLists.txt",
        """
set(CANN_ROOT "ignored" CACHE PATH "fixture")
set(OPBASE_INCLUDE_DIR "ignored" CACHE PATH "fixture")
set(OPBASE_LIBRARY "ignored" CACHE STRING "fixture")
add_library(opkit_runtime STATIC runtime.cpp)
add_library(opkit::runtime ALIAS opkit_runtime)
set_target_properties(opkit_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
""",
    )
    module = root / "aclnn/sample"
    write(
        module / "op_host/sample_tiling.cpp",
        'extern "C" int runtime_value();\n'
        'extern "C" int PrepareSample() { return runtime_value(); }',
    )
    write(
        module / "op_host/sample_binding.cpp",
        'extern "C" int SampleL0Contract() { return 0; }\n'
        'extern "C" int OpKitPrepareDirectInvokeSample() { return 0; }',
    )
    write(
        module / "op_api/aclnn_sample.h",
        'extern "C" int aclnnSample();\n'
        'extern "C" int aclnnSampleGetWorkspaceSize();\n',
    )
    if l0:
        write(
            module / "op_api/sample.cpp",
            'extern "C" int PrepareSample();\n'
            'extern "C" int SampleL0() { return PrepareSample(); }',
        )
    if l2:
        write(
            module / "op_api/aclnn_sample.cpp",
            'extern "C" int SampleL0();\n'
            'extern "C" int aclnnSample() { return SampleL0(); }\n'
            'extern "C" int aclnnSampleGetWorkspaceSize() { return 0; }',
        )
    write(module / "CMakeLists.txt", "# Generated OpKit module\n")
    write(
        root / "package.json",
        json.dumps(
            dict(
                kind="opkit",
                binary_format="device",
                mode="host",
                target="dav-3510",
                runtime="runtime",
                operators=["Sample"],
                modules={"Sample": "aclnn/sample"},
            )
        ),
    )
    write(
        root / "manifest.json",
        json.dumps(
            dict(
                files={
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in root.rglob("*")
                    if path.is_file()
                }
            )
        ),
    )


def publish_native(root):
    source = write(root / "native.cpp", 'extern "C" int native_entry() { return 43; }')
    obj = root.parent / "native.o"
    subprocess.run(["c++", "-fPIC", "-c", str(source), "-o", str(obj)], check=True)
    archive = root / "host/libsample.a"
    archive.parent.mkdir()
    subprocess.run(["ar", "rcs", str(archive), str(obj)], check=True)
    write(
        root / "CMakeLists.txt",
        "add_library(opkit_native_sample STATIC IMPORTED GLOBAL)\n"
        "set_target_properties(opkit_native_sample PROPERTIES IMPORTED_LOCATION "
        '"${CMAKE_CURRENT_LIST_DIR}/host/libsample.a")\n'
        'target_link_options(opkit_native_sample INTERFACE "LINKER:--undefined=native_entry")\n'
        "set_property(DIRECTORY PROPERTY _OPKIT_NATIVE_ARCHIVE_TARGETS opkit_native_sample)\n"
        "function(_opkit_native_finalize target)\n"
        "  get_target_property(links ${target} LINK_LIBRARIES)\n"
        '  if(NOT "ophost_transformer_cannbotdsl_tiling_obj" IN_LIST links)\n'
        '    message(FATAL_ERROR "Native objects missing from Host link")\n'
        "  endif()\n"
        '  file(WRITE "${CMAKE_BINARY_DIR}/native-finalized" "${target}")\n'
        "endfunction()\n",
    )


def verify_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["files"] == {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    return manifest


@pytest.mark.parametrize(
    "aclnn_type,l0,l2",
    [
        ("aclnn", True, True),
        ("aclnn_inner", True, False),
        ("aclnn_exclude", False, False),
    ],
)
def test_aclnn_publication_modes_cache_and_validation(
    builder, tmp_path, monkeypatch, aclnn_type, l0, l2
):
    calls = []

    def publish(**options):
        calls.append(options)
        publish_aclnn(options.pop("output"), **options)

    monkeypatch.setitem(sys.modules, "opkit", types.SimpleNamespace(build=publish))
    monkeypatch.setitem(
        sys.modules,
        "opkit.integrity",
        types.SimpleNamespace(verify_manifest=verify_manifest),
    )
    source = write(tmp_path / "sample.py", "# collector")
    output = tmp_path / "host"
    builder.build(source, output, "ascend950", "host", aclnn_type=aclnn_type)
    builder.build(source, output, "ascend950", "host", aclnn_type=aclnn_type)
    assert len(calls) == 1
    assert calls[0]["native"] is False and calls[0]["l0"] is l0 and calls[0]["l2"] is l2
    assert (output / "aclnn/sample/op_api/sample.cpp").is_file() == l0
    assert (output / "aclnn/sample/op_api/aclnn_sample.cpp").is_file() == l2
    # A failed replacement must leave the previous source package intact.
    previous = (output / "package.json").read_bytes()
    source.write_text("# changed collector")

    def invalid(**options):
        root = options.pop("output")
        publish_aclnn(root, **options)
        (root / "aclnn/sample/op_host/sample_binding.cpp").unlink()
        files = json.loads((root / "manifest.json").read_text())
        del files["files"]["aclnn/sample/op_host/sample_binding.cpp"]
        write(root / "manifest.json", json.dumps(files))

    monkeypatch.setattr(sys.modules["opkit"], "build", invalid)
    with pytest.raises(ValueError, match="Missing ACLNN module sources"):
        builder.build(source, output, "ascend950", "host", aclnn_type=aclnn_type)
    assert (output / "package.json").read_bytes() == previous
    assert (output / "aclnn/sample/op_host/sample_binding.cpp").is_file()


def aclnn_project(
    project, *, builtin, host, kernel, aclnn_type, native=True, aclnn=True
):
    # Stub only the publisher boundary; run the real cache, validation, CMake and linker.
    helper = project / "cmake/scripts/build_cannbotdsl.py"
    write(
        helper,
        f"""
import importlib.util, sys, types
from pathlib import Path
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
spec = importlib.util.spec_from_file_location('real_builder', {str(BUILDER)!r})
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
builder.framework_files = lambda: []
builder.source_exports = lambda source: dict(
    native='@aot.export' in source.read_text(),
    aclnn=any('@aclnn' in path.read_text() for path in (source, source.with_name('aclnn_' + source.name))
              if path.is_file()))
spec = importlib.util.spec_from_file_location('fixture', {str(Path(__file__).resolve())!r})
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
def publish(**options):
    if Path(options['collectors'], 'aclnn_sample.py').is_file():
        assert options['pattern'] == ('sample.py' if options['native'] else '*.py')
    with Path({str(project / "publications.log")!r}).open('a') as stream:
        kind = 'native' if options['native'] else 'aclnn'
        stream.write(kind + ':' + options['mode'] + '\\n')
    root = options.pop('output')
    if options['mode'] == 'kernel':
        assert options['native'] is True
        fixture.publish_aclnn(root, l0=False, l2=False, **options)
    elif options['native']:
        fixture.publish_native(root)
    else:
        fixture.publish_aclnn(root, **options)
sys.modules['opkit'] = types.SimpleNamespace(build=publish)
sys.modules['opkit.integrity'] = types.SimpleNamespace(verify_manifest=fixture.verify_manifest)
builder.main()
""",
    )
    shutil.copy2(
        ROOT / "cmake/aclnn_ops_transformer.h.in",
        project / "cmake/aclnn_ops_transformer.h.in",
    )
    write(
        project / "scripts/util/add_deprecation_warning.py",
        (ROOT / "scripts/util/add_deprecation_warning.py").read_text(),
    )
    write(
        project / "op_dsl/sample.py",
        ("# @aot.export Native collector\n" if native else "")
        + ("# @aclnn ACLNN entry\n" if aclnn else ""),
    )
    write(project / "op_dsl/shared input.py", "# input dependency")
    for module in ("cannbotdsl", "opkit"):
        write(project / "probe_modules" / (module + ".py"), "")
    write(
        project / "api.cpp",
        'extern "C" int asc_entry() { return 23; }\n'
        'extern "C" int SampleL0();\n'
        + (
            'extern "C" int PublicApi() { return SampleL0(); }'
            if aclnn_type == "aclnn_inner"
            else 'extern "C" int PublicApi() { return 29; }'
        ),
    )
    write(project / "asc.cpp", 'extern "C" int asc_host() { return 31; }')
    write(
        project / "CMakeLists.txt",
        f'''
cmake_minimum_required(VERSION 3.19)
project(aclnn_integration CXX)
include("{ROOT}/cmake/func.cmake")
include("{ROOT}/cmake/opbuild.cmake")
set(OPS_ADV_DIR "{project}")
set(HI_PYTHON "{sys.executable}")
set(ENV{{PYTHONPATH}} "{project}/probe_modules:$ENV{{PYTHONPATH}}")
set(ASCEND_CANN_PACKAGE_PATH "{project}/sdk")
set(ASCEND_COMPUTE_UNIT ascend950)
set(CANNBOTDSL_COMPUTE_UNIT ascend950)
set(CANNBOTDSL_TARGET dav-3510)
set(ASCEND_BINARY_OUT_DIR "${{CMAKE_BINARY_DIR}}/binary")
set(ASCEND_AUTOGEN_PATH "${{CMAKE_BINARY_DIR}}/autogen")
set(ENABLE_BUILT_IN {"ON" if builtin else "OFF"})
set(ENABLE_OPS_HOST {"ON" if host else "OFF"})
set(ENABLE_OPS_KERNEL {"ON" if kernel else "OFF"})
set(VENDOR_NAME review)
set(OPHOST_NAME ophost_transformer)
set(OPAPI_NAME opapi_transformer)
set(ACLNN_INC_INSTALL_DIR include)
set(ACLNN_INC_LEVEL2_INSTALL_DIR include/level2)
set(ASCEND_PYTHON_EXECUTABLE "{sys.executable}")
set(MODULE_OPTYPE sample)
set(MODULE_ACLNNTYPE {aclnn_type})
include("{ROOT}/cmake/symbol.cmake")
enable_cannbotdsl_kernel(sample "{project}/op_dsl")
cannbotdsl_clean_resources()
if(ENABLE_OPS_HOST)
  add_library(${{OPHOST_NAME}}_tiling_obj OBJECT asc.cpp)
  set_target_properties(${{OPHOST_NAME}}_tiling_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  foreach(dependency intf_pub_cxx17 c_sec register rt2_registry_static tiling_api runtime acl_rt mmpa)
    add_library(${{dependency}} INTERFACE)
  endforeach()
  if(ENABLE_BUILT_IN)
    set(OPHOST_LIB_INSTALL_PATH lib)
    gen_ophost_symbol()
  else()
    set(OPTILING_LIB_INSTALL_DIR lib)
    set(OPTILING_INSTALL_DIR lib)
    gen_cust_optiling_symbol()
    set_target_properties(cust_opmaster PROPERTIES OUTPUT_NAME cust_opmaster)
  endif()
endif()
add_custom_target(ops_transformer_config ALL)
set(ASCENDC_CMAKE_UTIL_DIR "{project}/indexer")
if(ENABLE_OPS_KERNEL)
  add_bin_compile_target(COMPUTE_UNIT ascend950)
endif()
if(ENABLE_OPS_HOST)
  set(CANN_ROOT "keep-root" CACHE PATH "caller root" FORCE)
  set(OPBASE_INCLUDE_DIR "keep-include" CACHE PATH "caller headers" FORCE)
  set(OPBASE_LIBRARY "keep-library" CACHE STRING "caller library" FORCE)
  foreach(kind aclnn aclnn_inner aclnn_exclude)
    add_library(${{OPHOST_NAME}}_opdef_${{kind}}_obj INTERFACE)
  endforeach()
  add_library(${{OPHOST_NAME}}_aclnn_exclude_headers INTERFACE)
  add_library(${{OPHOST_NAME}}_opapi_obj OBJECT api.cpp)
  set_target_properties(${{OPHOST_NAME}}_opapi_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  if(NOT ENABLE_BUILT_IN)
    add_library(cust_opapi SHARED api.cpp
      $<$<TARGET_EXISTS:${{OPHOST_NAME}}_cannbotdsl_opapi_obj>:$<TARGET_OBJECTS:${{OPHOST_NAME}}_cannbotdsl_opapi_obj>>)
    target_link_libraries(cust_opapi PRIVATE $<TARGET_NAME_IF_EXISTS:${{OPHOST_NAME}}_cannbotdsl_opapi_obj>)
    set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME opapi)
  endif()
  gen_aclnn_with_opdef()
  if(NOT CANN_ROOT STREQUAL "keep-root" OR NOT OPBASE_INCLUDE_DIR STREQUAL "keep-include"
     OR NOT OPBASE_LIBRARY STREQUAL "keep-library")
    message(FATAL_ERROR "Runtime configuration changed caller cache")
  endif()
  if(TARGET ${{OPHOST_NAME}}_cannbotdsl_opapi_obj)
    get_target_property(sources ${{OPHOST_NAME}}_cannbotdsl_opapi_obj SOURCES)
    file(WRITE "${{CMAKE_BINARY_DIR}}/sources.txt" "${{sources}}")
  endif()
  if(ENABLE_BUILT_IN)
    foreach(dependency intf_pub ops_aclnn nnopbase profapi ge_common_base ascend_dump ascendalog opapi_math)
      add_library(${{dependency}} INTERFACE)
    endforeach()
    include("{ROOT}/cmake/symbol.cmake")
    gen_opapi_symbol()
    set_target_properties(${{OPAPI_NAME}} PROPERTIES OUTPUT_NAME opapi)
    install(TARGETS ${{OPAPI_NAME}} LIBRARY DESTINATION lib)
  else()
    install(TARGETS cust_opapi LIBRARY DESTINATION lib)
  endif()
endif()
''',
    )
    # The fixture indexer verifies that the existing shared ASC/DSL scan directory is used.
    write(
        project / "indexer/ascendc_ops_config.py",
        """
import json, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('-p') + 1])
assert (root / 'sample/Sample.o').read_bytes() == b'device'
(root / 'binary_info_config.json').write_text('{}')
(root / 'relocatable_kernel_info_config.json').write_text('{}')
""",
    )


@pytest.mark.skipif(
    sys.platform != "linux", reason="OPP shared libraries use the GNU ELF linker"
)
@pytest.mark.parametrize("builtin", [False, True])
@pytest.mark.parametrize("host,kernel", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize(
    "aclnn_type,count", [("aclnn", 4), ("aclnn_inner", 3), ("aclnn_exclude", 2)]
)
def test_aclnn_build_link_and_install(
    tmp_path, builtin, host, kernel, aclnn_type, count
):
    aclnn_project(
        tmp_path, builtin=builtin, host=host, kernel=kernel, aclnn_type=aclnn_type
    )
    cmake("-S", tmp_path, "-B", tmp_path / "build")
    cmake("--build", tmp_path / "build")
    cmake("--install", tmp_path / "build", "--prefix", tmp_path / "installed")
    calls = (tmp_path / "publications.log").read_text().splitlines()
    assert calls.count("native:host") == int(host)
    assert calls.count("aclnn:host") == int(host)
    assert calls.count("native:kernel") == int(kernel)
    assert len(calls) == 2 * int(host) + int(kernel)
    if host:
        assert len((tmp_path / "build/sources.txt").read_text().split(";")) == count
        assert (tmp_path / "installed/include/aclnn_sample.h").is_file() == (
            aclnn_type == "aclnn"
        )
        master = tmp_path / (
            "installed/include/aclnn_ops_transformer.h"
            if builtin
            else "installed/include/aclnn_ops_transformer_review.h"
        )
        assert ('#include "aclnn_sample.h"' in master.read_text()) == (
            aclnn_type == "aclnn"
        )
        library = ctypes.CDLL(str(tmp_path / "installed/lib/libopapi.so"))
        native_library = tmp_path / (
            "installed/lib/libophost_transformer.so"
            if builtin
            else "installed/lib/libcust_opmaster.so"
        )
        native = ctypes.CDLL(str(native_library))
        assert native.native_entry() == 43 and native.asc_host() == 31
        assert (tmp_path / "build/native-finalized").read_text() == (
            "ophost_transformer" if builtin else "cust_opmaster"
        )
        assert not hasattr(library, "native_entry")
        assert not hasattr(native, "aclnnSample")
        assert library.asc_entry() == 23
        assert library.PublicApi() == (17 if aclnn_type == "aclnn_inner" else 29)
        if aclnn_type == "aclnn":
            assert library.aclnnSample() == 17
        symbols = subprocess.check_output(
            ["nm", "-D", "--defined-only", str(tmp_path / "installed/lib/libopapi.so")],
            text=True,
        )
        assert "PrepareSample" not in symbols and "SampleL0Contract" not in symbols
    kernels = list((tmp_path / "installed").rglob("Sample.o"))
    assert len(kernels) == int(kernel)
    assert (
        bool(list((tmp_path / "installed").rglob("binary_info_config.json"))) == kernel
    )
    assert not list((tmp_path / "installed").rglob("*.a"))
    if kernel:
        expected = (
            "opp/built-in/op_impl/ai_core/tbe/kernel/ascend950/ops_transformer/sample/Sample.o"
            if builtin
            else "packages/vendors/review_transformer/op_impl/ai_core/tbe/kernel/ascend950/sample/Sample.o"
        )
        assert kernels[0].relative_to(tmp_path / "installed").as_posix() == expected
    dependencies = (
        (tmp_path / "build/CMakeFiles/Makefile.cmake").read_text().splitlines()
    )
    references = {
        (tmp_path / "build" / line.strip()[1:-1]).resolve()
        for line in dependencies
        if line.strip().startswith('"') and line.strip().endswith('"')
    }
    assert (tmp_path / "op_dsl/shared input.py").resolve() in references


@pytest.mark.skipif(
    sys.platform != "linux", reason="OPP shared libraries use the GNU ELF linker"
)
@pytest.mark.parametrize("builtin", [False, True])
def test_dual_host_caches_are_independent_and_aclnn_runs_without_python(
    tmp_path, builtin, monkeypatch
):
    # Keep the compiler input stable across first configure and cached CMake runs.
    monkeypatch.setenv("CXX", shutil.which("c++"))
    aclnn_project(tmp_path, builtin=builtin, host=True, kernel=True, aclnn_type="aclnn")
    build = tmp_path / "build"
    cmake("-S", tmp_path, "-B", build)
    cmake("--build", build)
    cmake("--install", build, "--prefix", tmp_path / "installed")
    packages = build / "cannbot_dsl/sample"
    native_state = (packages / "host/.transformer-build.json").read_bytes()
    assert json.loads(native_state)["record"]["aclnn_type"] is None
    assert (
        json.loads((packages / "aclnn/host/.transformer-build.json").read_text())[
            "record"
        ]["aclnn_type"]
        == "aclnn"
    )
    initial = (tmp_path / "publications.log").read_text()
    cmake("-S", tmp_path, "-B", build)
    cmake("--build", build)
    assert (tmp_path / "publications.log").read_text() == initial
    # Damage only the ACLNN output: rebuilding it must not republish Native or devices.
    (packages / "aclnn/host/aclnn/sample/op_api/aclnn_sample.cpp").write_text("damaged")
    cmake("-S", tmp_path, "-B", build)
    cmake("--build", build)
    assert (tmp_path / "publications.log").read_text() == initial + "aclnn:host\n"
    assert (packages / "host/.transformer-build.json").read_bytes() == native_state
    main = write(
        tmp_path / "main.cpp",
        '#include "aclnn_sample.h"\nint main() { return aclnnSample() == 17 ? 0 : 1; }',
    )
    subprocess.run(
        [
            "c++",
            str(main),
            "-I" + str(tmp_path / "installed/include"),
            "-L" + str(tmp_path / "installed/lib"),
            "-lopapi",
            "-Wl,-rpath," + str(tmp_path / "installed/lib"),
            "-o",
            str(tmp_path / "consumer"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # A native executable calls the installed ACLNN ABI without a Python interpreter.
    subprocess.run(
        [str(tmp_path / "consumer")],
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "/absent"},
        check=True,
    )
    dynamic = subprocess.check_output(
        ["readelf", "-d", str(tmp_path / "installed/lib/libopapi.so")], text=True
    )
    needed = [line for line in dynamic.splitlines() if "NEEDED" in line]
    assert not any(
        name in line.lower()
        for line in needed
        for name in ("python", "cannbotdsl", "opkit")
    )


@pytest.mark.skipif(
    sys.platform != "linux", reason="OPP shared libraries use the GNU ELF linker"
)
@pytest.mark.parametrize("native,aclnn", [(True, False), (False, True)])
def test_reconfigure_removes_only_unselected_host_path(tmp_path, native, aclnn):
    aclnn_project(tmp_path, builtin=False, host=True, kernel=True, aclnn_type="aclnn")
    build = tmp_path / "build"
    cmake("-S", tmp_path, "-B", build)
    cmake("--build", build)
    source = tmp_path / "op_dsl/sample.py"
    source.write_text(
        ("# @aot.export Native collector\n" if native else "")
        + ("# @aclnn ACLNN entry\n" if aclnn else "")
    )
    cmake("-S", tmp_path, "-B", build)
    cmake("--build", build)
    # Inspect current link inputs rather than stale, unreferenced files from the old build.
    api_link = (build / "CMakeFiles/cust_opapi.dir/link.txt").read_text()
    native_link = (build / "CMakeFiles/cust_opmaster.dir/link.txt").read_text()
    assert ("cannbotdsl_opapi_obj" in api_link) == aclnn
    assert ("objects/libsample/native.o" in native_link) == native
    assert (build / "binary/ascend950/bin/sample/Sample.o").is_file()


@pytest.mark.skipif(
    sys.platform != "linux", reason="OPP shared libraries use the GNU ELF linker"
)
def test_separate_declaration_publishes_public_api_with_existing_asc_inner_registration(
    tmp_path,
):
    aclnn_project(
        tmp_path,
        builtin=False,
        host=True,
        kernel=True,
        aclnn_type="aclnn_inner",
        aclnn=False,
    )
    write(
        tmp_path / "op_dsl/aclnn_sample.py", "# @aclnn standalone public declaration\n"
    )
    cmake("-S", tmp_path, "-B", tmp_path / "build")
    cmake("--build", tmp_path / "build")
    cmake("--install", tmp_path / "build", "--prefix", tmp_path / "installed")
    state = json.loads(
        (
            tmp_path / "build/cannbot_dsl/sample/aclnn/host/.transformer-build.json"
        ).read_text()
    )
    assert state["record"]["aclnn_type"] == "aclnn"
    assert (tmp_path / "installed/include/aclnn_sample.h").is_file()
    library = ctypes.CDLL(str(tmp_path / "installed/lib/libopapi.so"))
    assert library.aclnnSample() == 17


def test_source_exports_use_framework_recognizer_with_separate_declaration(tmp_path):
    pytest.importorskip("cannbotdsl.aot.collection")
    spec = importlib.util.spec_from_file_location("separate_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = write(
        tmp_path / "sample.py",
        'raise RuntimeError("must not execute source")\n'
        "from cannbotdsl import aot as collector\n"
        '@collector.export("sample")\ndef collect(): pass\n',
    )
    declaration = write(
        tmp_path / "aclnn_sample.py",
        'from opkit import aclnn as api\n@api("Sample")\ndef entry(): pass\n',
    )
    assert module.source_exports(source) == dict(native=True, aclnn=True)
    # The explicit decorator controls DSL ACLNN/L2 generation; Native remains available.
    declaration.write_text(declaration.read_text().replace('@api("Sample")\n', ""))
    assert module.source_exports(source) == dict(native=True, aclnn=False)
    declaration.unlink()
    assert module.source_exports(source) == dict(native=True, aclnn=False)


@pytest.mark.skipif(
    sys.platform != "linux", reason="OPP shared libraries use the GNU ELF linker"
)
@pytest.mark.parametrize("builtin", [False, True])
def test_l2_declaration_changes_do_not_republish_native_or_device(
    tmp_path, builtin, monkeypatch
):
    monkeypatch.setenv("CXX", shutil.which("c++"))
    aclnn_project(
        tmp_path,
        builtin=builtin,
        host=True,
        kernel=True,
        aclnn_type="aclnn",
        native=True,
        aclnn=False,
    )
    declaration = tmp_path / "op_dsl/aclnn_sample.py"
    build = tmp_path / "build"
    packages = build / "cannbot_dsl/sample"
    snapshots = None
    for state in ("absent", "added", "modified", "deleted"):
        if state in ("added", "modified"):
            write(declaration, "# @aclnn controlled L2 declaration: " + state)
        else:
            declaration.unlink(missing_ok=True)
        cmake("-S", tmp_path, "-B", build)
        cmake("--build", build)
        installed = tmp_path / ("installed-" + state)
        cmake("--install", build, "--prefix", installed)
        current = [
            (packages / name).read_bytes()
            for name in (
                "inputs.json",
                "host/.transformer-build.json",
                "device/.transformer-build.json",
            )
        ]
        current.append((packages / "kernel.built").stat().st_mtime_ns)
        if snapshots is None:
            snapshots = current
        assert current == snapshots
        publications = (tmp_path / "publications.log").read_text().splitlines()
        assert publications.count("native:host") == 1
        assert publications.count("native:kernel") == 1
        assert (
            publications.count("aclnn:host")
            == {"absent": 0, "added": 1, "modified": 2, "deleted": 2}[state]
        )
        native = installed / (
            "lib/libophost_transformer.so" if builtin else "lib/libcust_opmaster.so"
        )
        assert ctypes.CDLL(str(native)).native_entry() == 43
        assert len(list(installed.rglob("Sample.o"))) == 1
        assert (installed / "include/aclnn_sample.h").is_file() == (
            state in ("added", "modified")
        )
