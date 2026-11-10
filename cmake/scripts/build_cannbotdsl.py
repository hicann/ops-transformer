#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Build split DSL resources and describe the consumer's existing Host library."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


_NATIVE_ARCHIVE_INDEX = "host/config/archive_info_config.json"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def framework_files():
    """Watch the installed compiler/runtime, including editable source installs."""
    paths = set()
    try:
        from cannbotdsl.core.toolchain.ascendc import resolve_bisheng

        compiler = Path(resolve_bisheng().path).resolve()
        if compiler.is_file():
            paths.add(compiler)
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError):
        pass
    for name, module in (("cannbot-dsl", "cannbotdsl"), ("opkit", "opkit")):
        package = import_module(module)
        roots = {Path(package.__file__).resolve().parent}
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            dist = None
        if dist is not None:
            roots.add(Path(dist.locate_file(module)).resolve())
        for root in roots:
            paths.update(
                p
                for p in root.rglob("*")
                if p.is_file()
                and (
                    p.suffix in (".py", ".so", ".h", ".cpp", ".in")
                    or p.name in ("CMakeLists.txt", "opkit-opt")
                )
            )
        if dist is not None:
            paths.update(
                Path(dist.locate_file(p)).resolve()
                for p in (dist.files or ())
                if p.name in ("RECORD", "METADATA")
            )
    sdk = Path(os.environ["ASCEND_HOME_PATH"])
    paths.update(
        p for p in (sdk / "version.info", sdk / "compiler/version.info") if p.is_file()
    )
    return sorted(paths)


def source_exports(source):
    # Use the same alias-aware source recognizer as OpKit; do not execute Python here.
    from cannbotdsl.aot.collection import _declares_registration

    adapter = source.with_name("aclnn_" + source.name)
    return dict(
        native=_declares_registration(source, {"export"}),
        aclnn=any(
            _declares_registration(path, {"aclnn"})
            for path in (source, adapter)
            if path.is_file()
        ),
    )


def input_record(source, soc, aclnn_type=None, target="dav-3510"):
    source = Path(source).resolve(strict=True)
    files = {*source.parent.rglob("*.py"), *framework_files(), Path(__file__).resolve()}
    exports = source_exports(source)
    if aclnn_type is None:
        files.discard(source.with_name("aclnn_" + source.name))
        exports = dict(native=exports["native"])
    return dict(
        schema=1,
        source=str(source),
        soc=soc,
        target=target,
        python=sys.executable,
        sdk=os.environ.get("ASCEND_HOME_PATH"),
        environment={
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(("CANNBOTDSL_", "CANNBOT_OPKIT_")) or k in ("CXX", "AR")
        },
        files={str(p): digest(p) for p in sorted(files)},
        exports=exports,
    )


def write_json(path, value):
    path = Path(path)
    raw = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if not path.is_file() or path.read_text() != raw:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(raw)
        temporary.replace(path)


@contextmanager
def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@contextmanager
def publication(output):
    """Serialize replacement and retain the previous package on build failure."""
    output = Path(output)
    with file_lock(output.with_name(output.name + ".lock")):
        with tempfile.TemporaryDirectory(
            prefix=".dsl-", dir=output.parent
        ) as directory:
            stage = Path(directory) / "package"
            yield stage
            backup = Path(directory) / "previous"
            if output.exists():
                output.rename(backup)
            try:
                stage.rename(output)
            except BaseException:
                if backup.exists():
                    backup.rename(output)
                raise


def package_files(root):
    return {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != ".transformer-build.json"
    }


def sync_kernels(output, destination, soc):
    """Own individual scan files; never delete another producer's directory."""
    destination = Path(destination)
    state = destination.parent / ".cannbotdsl-files.json"
    with file_lock(state.with_suffix(".lock")):
        _sync_kernels(output, destination, soc, state)


def remove_scan_files(root, paths):
    directories = set()
    for path in paths:
        file = root / path
        file.unlink(missing_ok=True)
        directories.add(file.parent)
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        while directory != root:
            try:
                directory.rmdir()
            except OSError:
                break
            if directory.parent == root:
                (root / (directory.name + ".json")).unlink(missing_ok=True)
            directory = directory.parent


def same_resource(source, target):
    if digest(source) == digest(target):
        return True
    if source.suffix == ".json":
        original, scanned = (
            json.loads(source.read_text()),
            json.loads(target.read_text()),
        )
        # The standard ASC indexer changes only this field in the scan copy.
        original.pop("filePath", None)
        scanned.pop("filePath", None)
        return original == scanned
    return False


def _sync_kernels(output, destination, soc, state):
    owners = json.loads(state.read_text()) if state.is_file() else {}
    owner = str(output)
    resources = output / "kernel" / soc
    current = [
        p.relative_to(resources).as_posix()
        for p in sorted(resources.rglob("*"))
        if p.is_file()
    ]
    other = {p for name, paths in owners.items() if name != owner for p in paths}
    if other.intersection(current):
        raise ValueError("DSL packages publish conflicting device paths")
    previous = set(owners.get(owner, []))
    for path in current:
        target = destination / path
        if (
            target.is_file()
            and path not in previous
            and not same_resource(resources / path, target)
        ):
            raise ValueError(f"Refusing to replace an unowned scan file: {target}")
    remove_scan_files(destination, set(owners.get(owner, [])) - set(current))
    for path in current:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or digest(target) != digest(resources / path):
            shutil.copy2(resources / path, target)
    owners[owner] = current
    write_json(state, owners)


def clean(root, binary_root, packages, host_enabled):
    root, binary_root = Path(root), Path(binary_root)
    selected = {str(Path(p).absolute()) for p in packages}
    # Adopt the previous layout's known files when upgrading an existing build.
    for destination in binary_root.glob("*/bin"):
        state = destination.parent / ".cannbotdsl-files.json"
        if state.is_file():
            continue
        owners = {}
        for resources in root.glob("*/device/kernel/" + destination.parent.name):
            owner = str(resources.parent.parent.absolute())
            owners[owner] = [
                p.relative_to(resources).as_posix()
                for p in resources.rglob("*")
                if p.is_file()
                and (destination / p.relative_to(resources)).is_file()
                and same_resource(p, destination / p.relative_to(resources))
            ]
        if owners:
            write_json(state, owners)
    for state in binary_root.glob("*/.cannbotdsl-files.json"):
        owners = json.loads(state.read_text())
        removed = set(owners) - selected
        if not removed:
            continue
        for owner in removed:
            remove_scan_files(state.parent / "bin", owners.pop(owner))
            (Path(owner).parent / "kernel.built").unlink(missing_ok=True)
        for name in ("binary_info_config.json", "relocatable_kernel_info_config.json"):
            (state.parent / "bin" / name).unlink(missing_ok=True)
        write_json(state, owners)
    if not host_enabled:
        shutil.rmtree(root / "metadata", ignore_errors=True)


def aclnn_required_files(module, aclnn_type):
    stem = Path(module).name
    files = [
        f"op_api/aclnn_{stem}.h",
        f"op_host/{stem}_tiling.cpp",
        f"op_host/{stem}_binding.cpp",
    ]
    if aclnn_type != "aclnn_exclude":
        files.append(f"op_api/{stem}.cpp")
    if aclnn_type == "aclnn":
        files.append(f"op_api/aclnn_{stem}.cpp")
    return [f"{module}/{file}" for file in files]


def validate_aclnn_host(root, aclnn_type, target="dav-3510"):
    """Check the generated ACLNN sources before replacing a published package."""
    from opkit.integrity import verify_manifest

    files = verify_manifest(root)["files"]
    package = json.loads((root / "package.json").read_text())
    if (
        package.get("kind") != "opkit"
        or package.get("binary_format") != "device"
        or package.get("mode") != "host"
        or package.get("target") != target
        or package.get("runtime") != "runtime"
        or "runtime/CMakeLists.txt" not in files
        or not package.get("operators")
        or set(package["operators"]) != set(package.get("modules", {}))
    ):
        raise ValueError("Expected an OpKit ACLNN Host source package")
    for name, module in package["modules"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or not re.fullmatch(
            r"aclnn/[a-z][a-z0-9_]*", module
        ):
            raise ValueError("Invalid ACLNN module name or path")
        required = [
            f"{module}/CMakeLists.txt",
            *aclnn_required_files(module, aclnn_type),
        ]
        if any(path not in files for path in required):
            raise ValueError(f"Missing ACLNN module sources: {name}")
    return package


def collector_pattern(source, aclnn_type):
    if aclnn_type is not None and source.with_name("aclnn_" + source.name).is_file():
        return "*.py"
    return source.name


def host_layout(source, output, aclnn_type):
    from cannbotdsl.aot.collection import Mark, discover
    from opkit.api import _compilation_symbols
    from opkit.package import Package, _native_host_layout

    package = Package(output.parent)

    def describe(entry):
        if entry.name not in package.modules:
            package.add(entry.name, {})
        return lambda: None

    mark = Mark(frozenset({"aclnn"}), "__cannbot_aclnn__", describe)
    discovery = discover(
        collectors=source.parent,
        pattern=collector_pattern(source, aclnn_type),
        marks=[] if aclnn_type is None else [mark],
        exclude_dirs=frozenset({"tests", "test", "build", "dist"}),
    )
    with _compilation_symbols(), discovery as tasks:
        if aclnn_type is None:
            _native_host_layout(tasks, output.parent)
            archives = {task: f"host/lib{task}.a" for task in tasks}
            files = [_NATIVE_ARCHIVE_INDEX]
            for archive in archives.values():
                files.extend((archive, Path(archive).with_suffix(".json").as_posix()))
            write_json(output, dict(archives=archives, files=files))
        else:
            if not package.modules:
                raise ValueError(f"No ACLNN entry found in {source}")
            runtime = distribution("opkit").locate_file("opkit/resources/runtime")
            files = []
            sources = {}
            for name, module in package.modules.items():
                stem = Path(module).name
                required = aclnn_required_files(module, aclnn_type)
                sources[name] = [path for path in required if path.endswith(".cpp")]
                files.extend(required)
                files.extend(
                    (f"{module}/op_api/{stem}.h", f"{module}/op_host/{stem}_tiling.h")
                )
            write_json(
                output,
                dict(
                    aclnn_type=aclnn_type,
                    operators=list(package.modules),
                    modules=package.modules,
                    runtime=str(runtime),
                    sources=sources,
                    files=files,
                ),
            )


def build(
    source,
    output,
    soc,
    mode,
    kernel_output=None,
    inputs=None,
    aclnn_type=None,
    target="dav-3510",
    layout=None,
    jobs=None,
):
    import opkit

    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    record = dict(
        inputs=json.loads(Path(inputs).read_text())
        if inputs
        else input_record(source, soc, aclnn_type, target),
        mode=mode,
        aclnn_type=aclnn_type,
    )
    if mode == "host":
        plan = json.loads(Path(layout).read_text())
        record["layout"] = plan
    state = output / ".transformer-build.json"
    cached = json.loads(state.read_text()) if state.is_file() else {}
    if cached.get("record") != record or cached.get("files") != package_files(output):
        with publication(output) as stage:
            options = (
                dict(native=True)
                if aclnn_type is None
                else dict(
                    native=False,
                    l0=aclnn_type != "aclnn_exclude",
                    l2=aclnn_type == "aclnn",
                )
            )
            opkit.build(
                collectors=source.parent,
                # Only ACLNN Host publication collects the L2 declaration.
                # Native Host and device profiles come from the implementation.
                pattern=collector_pattern(source, aclnn_type),
                target=target,
                output=stage,
                mode=mode,
                jobs=jobs,
                **options,
            )
            if mode == "host":
                if aclnn_type is None:
                    index = json.loads((stage / _NATIVE_ARCHIVE_INDEX).read_text())
                    actual = {
                        task: item["binPath"] for task, item in index["tasks"].items()
                    }
                    if plan["archives"] != actual:
                        raise ValueError(
                            "Native archive paths differ from the configured layout"
                        )
                else:
                    package = validate_aclnn_host(stage, aclnn_type, target)
                    if plan["modules"] != package["modules"]:
                        raise ValueError(
                            "ACLNN modules differ from the configured layout"
                        )
                if any(not (stage / path).is_file() for path in plan["files"]):
                    raise ValueError(
                        "Host package is missing files declared in the configured layout"
                    )
            write_json(
                stage / ".transformer-build.json",
                dict(record=record, files=package_files(stage)),
            )
    if mode == "kernel":
        # Feed the same ASC index generator; retain logical SoC paths in JSON.
        sync_kernels(output, kernel_output, soc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("host", "kernel", "inputs", "layout", "clean"), required=True
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--soc", default="ascend950")
    parser.add_argument("--target", default="dav-3510")
    parser.add_argument("--kernel-output", type=Path)
    parser.add_argument("--packages", nargs="*", type=Path, default=[])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--jobs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--host-enabled", action="store_true")
    parser.add_argument(
        "--aclnn-type", choices=("aclnn", "aclnn_inner", "aclnn_exclude")
    )
    args = parser.parse_args()
    if args.output is None:
        parser.error("--output is required for publication modes")
    if args.mode == "host" and args.layout is None:
        parser.error("--layout is required for Host publication")
    if args.mode == "inputs":
        record = input_record(args.source, args.soc, args.aclnn_type, args.target)
        write_json(args.output, record)
        args.output.with_suffix(".list").write_text(
            "\n".join(record["files"]) + "\n", encoding="utf-8"
        )
        args.output.with_suffix(".cmake").write_text(
            "".join(
                f"set(_dsl_{name} {'TRUE' if enabled else 'FALSE'})\n"
                for name, enabled in source_exports(args.source).items()
            )
        )
    elif args.mode == "layout":
        host_layout(args.source, args.output, args.aclnn_type)
    elif args.mode == "clean":
        clean(args.output, args.kernel_output, args.packages, args.host_enabled)
    else:
        build(
            args.source,
            args.output,
            args.soc,
            args.mode,
            args.kernel_output,
            args.inputs,
            args.aclnn_type,
            args.target,
            args.layout,
            args.jobs,
        )


if __name__ == "__main__":
    sys.exit(main())
