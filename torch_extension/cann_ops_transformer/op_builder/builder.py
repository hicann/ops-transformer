# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
from abc import ABC, abstractmethod
from typing import List, Union
import torch
from torch.utils.cpp_extension import load
from torch.library import Library

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
_as_library = None


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

    def _ensure_initialized(self):
        if self._initialized:
            return
        import torch_npu

        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        self._package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._cann_path = self.get_cann_path()
        if not hasattr(torch.ops.cann_ops_transformer, self.name):
            self.register_schema(self.schema())
            self.register_meta()
        self._initialized = True

    def ensure_initialized(self):
        self._ensure_initialized()

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ and os.path.exists(
            os.environ[ASCEND_HOME_PATH]
        ):
            return os.environ[ASCEND_HOME_PATH]
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
        paths.extend(
            [
                os.path.join(self._torch_npu_path, "include"),
                os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
                os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
                os.path.join(self._torch_npu_path, "include/third_party/op-plugin"),
                os.path.join(self._cann_path, "include"),
                os.path.join(self._cann_path, "include/aclnnop"),
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

        op_module = load(
            name=self.name,
            sources=self.get_absolute_paths(self.sources()),
            extra_include_paths=self.get_absolute_paths(self.include_paths()),
            extra_cflags=self.cxx_args(),
            extra_ldflags=self.extra_ldflags(),
            verbose=verbose,
        )
        OpBuilder._loaded_ops[self.name] = op_module

        return op_module
