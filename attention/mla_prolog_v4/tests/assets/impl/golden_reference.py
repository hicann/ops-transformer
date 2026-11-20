# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""重导出本地 CPU Golden，保持与 pytest 测试框架解耦。"""

import importlib.util
import sys
from pathlib import Path

_NAME = "mla_prolog_v4_pytest_9eeace4"
if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _NAME, Path(__file__).with_name("pytest_reference.py")
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)
_ref = sys.modules[_NAME]
globals().update({k: v for k, v in vars(_ref).items() if not k.startswith("__")})

mla_prolog_v4 = _ref.golden_mla_prolog
