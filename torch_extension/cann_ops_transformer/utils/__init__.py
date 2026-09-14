# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from .arg_check import (
    check_args,
    wrap_op_module,
    check_value,
    require_bool,
    require_dtype,
    require_float,
    require_int,
    require_list_tensor,
    require_optional_dtype,
    require_optional_float,
    require_optional_int,
    require_optional_list_tensor,
    require_optional_str,
    require_optional_tensor,
    require_str,
    require_tensor,
)

__all__ = [
    "check_args",
    "wrap_op_module",
    "check_value",
    "require_bool",
    "require_dtype",
    "require_float",
    "require_int",
    "require_list_tensor",
    "require_optional_dtype",
    "require_optional_float",
    "require_optional_int",
    "require_optional_list_tensor",
    "require_optional_str",
    "require_optional_tensor",
    "require_str",
    "require_tensor",
]
