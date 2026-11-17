# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Share the INT8 reference with the repository TestSpec assets."""

import importlib.util
import sys
from pathlib import Path

_NAME = "quant_flash_attn_int8_golden"
if _NAME not in sys.modules:
    _path = Path(__file__).resolve().parents[3] / "assets" / f"{_NAME}.py"
    _spec = importlib.util.spec_from_file_location(_NAME, _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)

golden = sys.modules[_NAME]
attention_reference = golden.attention_reference
assert_atk_accuracy = golden.assert_atk_accuracy
_inputs = golden._inputs
_pack = golden._pack
