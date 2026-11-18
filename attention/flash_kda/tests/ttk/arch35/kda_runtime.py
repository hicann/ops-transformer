# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""arch35 的 CANN 包加载适配；显式源码和二进制入口复用共享加载器。"""

import hashlib
import importlib
import importlib.metadata
import os
import pkgutil
from functools import lru_cache
from pathlib import Path

import kda_runtime as shared_runtime
from kda_cases import architecture


_loaded = None
_identity = None


def selection():
    kind = os.environ.get("KDA_DUT", "package")
    if kind not in ("package", "installed", "source"):
        raise ValueError("KDA_DUT must be package, installed, or source")
    arch = architecture()
    if arch != "arch35":
        raise RuntimeError("arch35 adapter requires FLASH_KDA_ARCH=arch35")
    root = Path(os.environ.get("KDA_SOURCE_ROOT", Path(__file__).resolve().parents[5]))
    return kind, arch, str(root.resolve())


def _package_candidates():
    preferred = (
        "cann_ops_transformer",
        "cann_ops_transformer_custom",
        "cann_ops_transformer_flash_kda",
    )
    for name in preferred:
        yield f"{name}.ops.attention.flash_kda"
    # 常用包均未命中时，再查找其他供应商前缀的已安装包。
    for name in sorted({item.name for item in pkgutil.iter_modules()}):
        if name.startswith("cann_ops_transformer") and name not in preferred:
            yield f"{name}.ops.attention.flash_kda"


def load_cann_package():
    attempted = []
    for name in _package_candidates():
        attempted.append(name)
        try:
            return importlib.import_module(name), name
        except ModuleNotFoundError as exc:
            # 仅跳过不存在的目标包；包内部依赖缺失时保留原异常，避免误选其他包。
            if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
                continue
            raise
    raise ModuleNotFoundError(
        "Cannot find FlashKDA in the CANN Python packages; "
        "source the CANN environment and install its Python package. Tried: "
        + ", ".join(attempted)
    )


def load():
    global _loaded, _identity
    identity = selection()
    if _loaded is not None:
        if _identity != identity:
            raise RuntimeError("DUT source/architecture changed; start a fresh process")
        return _loaded
    kind, arch, root = identity
    if kind != "package":
        # 保留共享加载器对 source 路径和 installed 二进制证据的检查。
        loaded = shared_runtime.load()
    else:
        # 默认 JIT 编译 CANN 包内的 Python；调用者显式设置的模式保持不变。
        os.environ.setdefault("CANNBOTDSL_NATIVE_BINARY_MODE", "off")
        public, package = load_cann_package()
        package_dir = Path(public.__file__).resolve().parent
        if package_dir.is_relative_to(Path(root)):
            raise RuntimeError(
                f"CANN package resolves inside source checkout: {package_dir}"
            )
        internal = tuple(
            importlib.import_module(f"{package}.op_dsl.{arch}.{name}")
            for name in ("flash_kda", "flash_kda_metadata", "flash_kda_validation")
        )
        for module in internal:
            expected = package_dir / "op_dsl" / arch
            if Path(module.__file__).resolve().parent != expected:
                raise RuntimeError(
                    "DUT implementation is outside the selected package: "
                    f"{module.__file__}"
                )
        loaded = public, internal
    _loaded, _identity = loaded, identity
    return _loaded


def public_api():
    return load()[0]


def modules():
    return load()[1]


@lru_cache(None)
def describe():
    public, internal = load()
    if selection()[0] != "package":
        return shared_runtime.describe()
    versions = {}
    for name in ("torch", "torch-npu", "cannbot-dsl", "cann-ops-transformer"):
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
        arch=selection()[1],
        public_path=public.__file__,
        implementation_path=internal[0].__file__,
        native_mode=os.environ.get("CANNBOTDSL_NATIVE_BINARY_MODE"),
        native_report=os.environ.get("CANNBOTDSL_NATIVE_BINARY_REPORT"),
        versions=versions,
        implementation_sha256=digest.hexdigest(),
    )
