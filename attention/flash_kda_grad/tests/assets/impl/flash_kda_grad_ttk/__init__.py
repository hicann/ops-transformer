# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may obtain a copy of the License at
# https://gitcode.com/cann/cannbot-dsl/blob/master/LICENSE.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

"""TTK adapter for ``flash_kda_grad``.

``flash_kda_grad_ttk.npu`` is the E2E ``api_name`` used by ``spec.py``;
``flash_kda_grad_ttk.golden`` holds the two CPU implementations that back the
TestSpec ``golden`` (FP64 autograd) and ``third_party`` (BF16 staged peer).
"""

from . import golden
from .api import npu

__all__ = ["npu", "golden"]
