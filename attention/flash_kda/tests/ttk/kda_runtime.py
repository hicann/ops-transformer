# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Explicit source/installed DUT selection and native-artifact evidence."""

import argparse
from functools import lru_cache
import hashlib
import importlib
import importlib.util
from importlib.machinery import ModuleSpec
import json
import logging
import time
import os
from pathlib import Path
import sys

from kda_cases import architecture


_loaded = None
_identity = None


def selection():
    kind = os.environ.get("KDA_DUT", "installed")
    if kind not in ("installed", "source"):
        raise ValueError("KDA_DUT must be installed or source")
    return (
        kind,
        architecture(),
        str(
            Path(
                os.environ.get("KDA_SOURCE_ROOT", Path(__file__).resolve().parents[4])
            ).resolve()
        ),
    )


def load():
    global _loaded, _identity
    identity = selection()
    if _loaded is not None:
        if _identity != identity:
            raise RuntimeError(
                "DUT source/architecture changed in one process; start a fresh process"
            )
        return _loaded
    kind, arch, root = identity
    if kind == "installed":
        mode = os.environ.setdefault("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
        if mode != "require":
            raise RuntimeError(
                "installed tests require CANNBOTDSL_NATIVE_BINARY_MODE=require"
            )
        if "CANNBOTDSL_NATIVE_BINARY_REPORT" not in os.environ:
            evidence = os.environ.get("KDA_EVIDENCE_DIR")
            if not evidence:
                raise RuntimeError(
                    "installed tests require KDA_EVIDENCE_DIR outside the checkout"
                )
            if (
                Path(evidence).resolve() == Path(root)
                or Path(root) in Path(evidence).resolve().parents
            ):
                raise RuntimeError("KDA_EVIDENCE_DIR must be outside the checkout")
            Path(evidence).mkdir(parents=True, exist_ok=True)
            os.environ["CANNBOTDSL_NATIVE_BINARY_REPORT"] = str(
                Path(evidence) / f"native-{os.getpid()}.json"
            )
        if any(n.startswith("attention.flash_kda.op_dsl.") for n in sys.modules):
            raise RuntimeError(
                "source DUT already imported; run installed tests in a fresh process"
            )
        public = importlib.import_module("cann_ops_transformer_flash_kda")
        package = "cann_ops_transformer_flash_kda.ops.attention.flash_kda"
        location = Path(public.__file__).resolve()
        if Path(root) in location.parents:
            raise RuntimeError(
                f"installed DUT resolves inside source checkout: {location}"
            )
    else:
        if "cann_ops_transformer_flash_kda" in sys.modules:
            raise RuntimeError(
                "installed DUT already imported; run source tests in a fresh process"
            )
        sys.path.insert(0, root)
        # Load the real builder without auto-discovering unrelated operators.
        if "cann_ops_transformer" not in sys.modules:
            parent = ModuleSpec("cann_ops_transformer", loader=None, is_package=True)
            parent.submodule_search_locations = [
                str(Path(root) / "torch_extension/cann_ops_transformer")
            ]
            sys.modules[parent.name] = importlib.util.module_from_spec(parent)
        package = "attention.flash_kda"
        importlib.import_module(package)
        name = package + "._ttk_public"
        path = Path(root) / "attention/flash_kda/torch_extension/flash_kda.py"
        spec = importlib.util.spec_from_file_location(name, path)
        public = importlib.util.module_from_spec(spec)
        sys.modules[name] = public
        spec.loader.exec_module(public)
    modules = tuple(
        importlib.import_module(f"{package}.op_dsl.{arch}.{name}")
        for name in ("flash_kda", "flash_kda_metadata", "flash_kda_validation")
    )
    if kind == "source":
        for module in modules:
            if (
                Path(module.__file__).resolve().parent
                != Path(root) / f"attention/flash_kda/op_dsl/{arch}"
            ):
                raise RuntimeError(
                    f"DUT imported from another checkout: {module.__file__}"
                )
    _loaded, _identity = (public, modules), identity
    return _loaded


def event(c, **values):
    folder = os.environ.get("KDA_EVIDENCE_DIR")
    if not folder:
        logging.info("KDA %s %s", c["id"], json.dumps(values, ensure_ascii=False))
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{c['id']}.{os.getpid()}.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                dict(case_id=c["id"], time_ns=time.time_ns(), **values),
                ensure_ascii=False,
            )
            + "\n"
        )


def public_api():
    return load()[0]


def modules():
    return load()[1]


@lru_cache(None)
def describe():
    public, internal = load()
    import importlib.metadata

    versions = {}
    for name in ("torch", "torch-npu", "cannbot-dsl", "cann-ops-transformer-flash-kda"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "distribution metadata unavailable"
    digest = hashlib.sha256()
    for path in sorted(Path(internal[0].__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return dict(
        dut=selection()[0],
        arch=architecture(),
        public_path=public.__file__,
        implementation_path=internal[0].__file__,
        native_mode=os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE"),
        native_report=os.environ.get("CANNBOTDSL_NATIVE_BINARY_REPORT"),
        versions=versions,
        source_commit=os.environ.get("KDA_SOURCE_COMMIT", "not recorded"),
        implementation_sha256=digest.hexdigest(),
    )


def check_native_reports(folder, *, require_hit=True):
    """Run after workers exit, when the SDK has flushed its native reports."""
    paths = list(Path(folder).rglob("native-*.json"))
    if not paths and require_hit:
        raise AssertionError("no SDK native reports; installed execution is unverified")
    totals = {"installed_hit": 0, "installed_miss": 0}

    def count(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in totals and isinstance(item, (int, float)):
                    totals[key] += item
                elif isinstance(item, (dict, list)):
                    count(item)
        elif isinstance(value, list):
            for item in value:
                count(item)

    for path in paths:
        count(json.loads(path.read_text(encoding="utf-8")))
    if (require_hit and totals["installed_hit"] <= 0) or totals["installed_miss"] != 0:
        raise AssertionError(f"installed native evidence failed: {totals}")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-native", required=True)
    print(check_native_reports(parser.parse_args().check_native))
