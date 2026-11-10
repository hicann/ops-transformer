# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import fcntl
import glob
import hashlib
import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import List, Union

import torch
from torch.library import Library
from torch.utils.cpp_extension import load

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
_as_library = None

logger = logging.getLogger(__name__)


def get_as_library():
    global _as_library
    if _as_library is None:
        try:
            _as_library = Library("cann_ops_transformer", "DEF")
        except RuntimeError:
            _as_library = Library("cann_ops_transformer", "FRAGMENT")
    return _as_library


class OpBuilder(ABC):
    _loaded_ops = {}

    def __init__(self, name, category=None):
        self.name = name
        self.category = category
        self._initialized = False

    def _is_op_registered(self):
        """Return True if ``self.name`` is already in torch.ops.cann_ops_transformer.

        Names may be nested (e.g. ``ds41.indexer_prologue_qw``). ``hasattr(ns, "a.b")``
        is always False for such names, so walk each segment instead. Matching the
        dispatcher layout avoids re-``define`` after a failed import left the schema
        registered but dropped the OpBuilder instance.
        """
        obj = torch.ops.cann_ops_transformer
        for part in self.name.split("."):
            if not hasattr(obj, part):
                return False
            obj = getattr(obj, part)
        return True

    def _ensure_initialized(self):
        if self._initialized:
            return
        import torch_npu

        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        self._package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._cann_path = self.get_cann_path()
        if not self._is_op_registered():
            self.register_schema(self.schema())
            self.register_meta()
        self._initialized = True

    def ensure_initialized(self):
        self._ensure_initialized()

    def get_cann_path(self):
        cann_path = os.environ.get(ASCEND_HOME_PATH)
        if cann_path and os.path.exists(cann_path):
            return cann_path
        return os.path.dirname(os.path.dirname(self._torch_npu_path))

    @property
    def cann_path(self):
        self._ensure_initialized()
        return self._cann_path

    @property
    def torch_npu_path(self):
        self._ensure_initialized()
        return self._torch_npu_path

    def get_absolute_paths(self, paths):
        self._ensure_initialized()
        return [os.path.join(self._package_path, path) for path in paths]

    def resolve_source(self, cpp_filename):
        self._ensure_initialized()
        if self.category is None:
            return cpp_filename
        return f"csrc/{self.category}/{cpp_filename}"

    def register_schema(self, op_schema: Union[str, List[str], None]):
        if op_schema is None:
            return
        if isinstance(op_schema, str):
            op_schema = [op_schema]
        for schema in op_schema:
            get_as_library().define(schema)

    @abstractmethod
    def sources(self): ...

    @abstractmethod
    def schema(self): ...

    @abstractmethod
    def register_meta(self): ...

    def _custom_opp_paths(self):
        custom_opp_paths = []
        custom_opp_env = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        for custom_opp_path in custom_opp_env.split(os.pathsep):
            custom_opp_path = custom_opp_path.strip()
            if custom_opp_path and os.path.isdir(custom_opp_path):
                custom_opp_paths.append(custom_opp_path)
        vendors_dir = os.path.join(self._cann_path, "opp", "vendors")
        if os.path.isdir(vendors_dir):
            for vendor_name in sorted(os.listdir(vendors_dir)):
                vendor_dir = os.path.join(vendors_dir, vendor_name)
                if os.path.isdir(vendor_dir):
                    custom_opp_paths.append(vendor_dir)
        return custom_opp_paths

    def include_paths(self):
        self._ensure_initialized()
        paths = []
        for vendor_dir in self._custom_opp_paths():
            inc = os.path.join(vendor_dir, "op_api", "include")
            if os.path.isdir(inc):
                paths.append(inc)
        # CANN 路径必须先于 torch_npu 路径：torch_npu 自带的 third_party/acl 版本较旧，
        # 且与新版 CANN 头文件共用同名 include guard，若排在前面会因 -I 搜索顺序先命中旧头，
        # 环境上实际安装的 CANN 头文件（含 ACL_FLOAT8_E5M2 等新枚举）被 guard 抑制导致编译报错。
        # vendor op_api/include 保持最优先，run 包的增强算子 aclnn 头文件需覆盖 CANN 同名头。
        paths.extend(
            [
                os.path.join(self._cann_path, "include"),
                os.path.join(self._cann_path, "include/aclnnop"),
                os.path.join(self._torch_npu_path, "include"),
                os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
                os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
                os.path.join(self._torch_npu_path, "include/third_party/op-plugin"),
                os.path.join(self._package_path, "common"),
            ]
        )
        return paths

    def cxx_args(self):
        args = [
            "-O3",
            "-w",
            "-fPIC",
            "-fstack-protector-all",
            "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-pie",
            "-s",
            "-fvisibility=hidden",
            "-D_FORTIFY_SOURCE=2",
        ]
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            args.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        else:
            args.append("-D_GLIBCXX_USE_CXX11_ABI=0")
        return args

    def extra_ldflags(self):
        self._ensure_initialized()
        flags = []
        for vendor_dir in self._custom_opp_paths():
            lib = os.path.join(vendor_dir, "op_api", "lib")
            if os.path.isdir(lib):
                flags.append("-L" + lib)
        flags.extend(
            [
                "-L" + os.path.join(self._cann_path, "lib64"),
                "-lascendcl",
                "-L" + os.path.join(self._torch_npu_path, "lib"),
                "-ltorch_npu",
            ]
        )
        return flags

    def load(self, verbose=True):
        self._ensure_initialized()
        if self.name in OpBuilder._loaded_ops:
            return OpBuilder._loaded_ops[self.name]

        ext_dir = os.path.expanduser(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR", os.path.join("~", ".cache", "torch_extensions")
            )
        )
        os.makedirs(ext_dir, exist_ok=True)
        lock_path = os.path.join(ext_dir, "%s.compile.lock" % self.name)
        fprint_path = os.path.join(ext_dir, "%s.compile.fprint" % self.name)

        # Lazy import avoids pulling arg_check into every builder import path.
        from ..utils.arg_check import wrap_op_module

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if self.name in OpBuilder._loaded_ops:
                return OpBuilder._loaded_ops[self.name]

            self._remove_stale_torch_lock()

            op_module = self._try_load_cached(wrap_op_module, fprint_path)
            if op_module is not None:
                return op_module

            op_module = load(
                name=self.name,
                sources=self.get_absolute_paths(self.sources()),
                extra_include_paths=self.get_absolute_paths(self.include_paths()),
                extra_cflags=self.cxx_args(),
                extra_ldflags=self.extra_ldflags(),
                verbose=verbose,
            )
            # Arg type-check proxy; ops can call through load() unchanged.
            op_module = wrap_op_module(op_module)
            OpBuilder._loaded_ops[self.name] = op_module
            self._record_sources_fingerprint(fprint_path)
            return op_module

    def _try_load_cached(self, wrap_op_module, fprint_path):
        """Reuse a freshly built extension instead of re-running the build back end.

        Called while holding the per-op compile lock. torch's build back end
        checks staleness by re-running ninja even when nothing changed; with
        many processes (e.g. the ranks of a distributed job) those no-op runs
        are serialised one after another, which dominates startup time. When
        the fingerprint recorded by the process that actually built these
        sources matches, importing the built extension is equivalent to the
        no-op rebuild and avoids the serialised checks.

        Only the op's source files are fingerprinted; header-only changes are
        not captured by design (set CANN_OP_BUILDER_NO_FPRINT_CACHE=1 to force
        the full build path in such cases). Any mismatch or lookup failure
        falls back to the full build path.
        """
        if os.environ.get("CANN_OP_BUILDER_NO_FPRINT_CACHE"):
            return None
        try:
            from torch.utils.cpp_extension import (
                _get_build_directory,
                _import_module_from_library,
            )

            fingerprint = self._compute_sources_fingerprint()
            if not fingerprint:
                return None
            with open(fprint_path) as f:
                if f.read().strip() != fingerprint:
                    return None
            build_dir = _get_build_directory(self.name, False)
            so_path = os.path.join(build_dir, "%s.so" % self.name)
            if not os.path.isfile(so_path):
                return None
            op_module = _import_module_from_library(self.name, build_dir, True)
            op_module = wrap_op_module(op_module)
            OpBuilder._loaded_ops[self.name] = op_module
            return op_module
        except Exception:
            return None

    def _compute_sources_fingerprint(self):
        """SHA1 over the resolved source file contents.

        Returns None when a source cannot be resolved or read, so callers
        never skip a build they cannot fingerprint.
        """
        digest = hashlib.sha1()
        for src in self.get_absolute_paths(self.sources()):
            if not isinstance(src, str):
                return None
            candidates = []
            if os.path.isfile(src):
                candidates = [src]
            elif any(ch in src for ch in "*?["):
                candidates = sorted(glob.glob(src))
            if not candidates:
                return None
            for path in candidates:
                try:
                    with open(path, "rb") as f:
                        digest.update(os.path.realpath(path).encode())
                        digest.update(b"\0")
                        digest.update(f.read())
                except OSError:
                    return None
        return digest.hexdigest()

    def _record_sources_fingerprint(self, fprint_path):
        fingerprint = self._compute_sources_fingerprint()
        if not fingerprint:
            return
        try:
            with open(fprint_path, "w") as f:
                f.write(fingerprint)
        except OSError:
            logger.warning(
                "Failed to record source fingerprint %s: build cache reuse "
                "will be skipped",
                fprint_path,
            )

    def _remove_stale_torch_lock(self):
        """Drop torch's FileBaton lock left by a killed JIT compile process.

        The caller must hold the exclusive flock for this op, so the torch
        lock, if present, cannot belong to a live compile that went through
        this builder; it is either a leftover of a killed process or absent.
        """
        try:
            from torch.utils.cpp_extension import _get_build_directory

            build_dir = _get_build_directory(self.name, False)
        except Exception:
            return
        torch_lock = os.path.join(build_dir, "lock")
        if os.path.isdir(torch_lock) or os.path.isfile(torch_lock):
            logger.warning("Detected stale build lock: %s", torch_lock)
        if os.path.isdir(torch_lock):
            try:
                shutil.rmtree(torch_lock)
            except OSError as e:
                logger.warning(
                    "Failed to remove stale build lock %s: %s", torch_lock, e
                )
        elif os.path.isfile(torch_lock):
            try:
                os.remove(torch_lock)
            except OSError as e:
                logger.warning(
                    "Failed to remove stale build lock %s: %s", torch_lock, e
                )
