# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import glob

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

# torch_npu installation root, used to locate ACL headers.
_TORCH_NPU_ROOT = os.path.dirname(os.path.abspath(torch_npu.__file__))
_ACL_INC_DIR = os.path.join(_TORCH_NPU_ROOT, "include", "third_party", "acl", "inc")
_NINJA_ENABLED = os.environ.get("USE_NINJA") == "1"
_HERE = os.path.dirname(os.path.realpath(__file__))
_CSRC_DIR = os.path.join(_HERE, "custom_ops", "csrc")
_csrc_sources = glob.glob(os.path.join(_CSRC_DIR, "*.cpp"))

_custom_ops_ext = NpuExtension(
    name="custom_ops.custom_ops_lib",
    sources=_csrc_sources,
    extra_compile_args=["-I" + _ACL_INC_DIR],
)

setup(
    name="custom_ops",
    version="1.0",
    keywords="custom_ops",
    ext_modules=[_custom_ops_ext],
    package_data={"custom_ops": ["*.py", "*.so"]},
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=_NINJA_ENABLED)},
)
