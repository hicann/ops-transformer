# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Register custom ops for torch.ops.custom and torch_npu."""

import os
import pkgutil
import warnings

import torch
import torch_npu

from . import custom_ops_lib as custom_ops_lib  # noqa: F401
from .minimax_sparse_attention_split_kv_csr import build_k2q_csr
from .npu_minimax_sparse_attention_split_kv import npu_minimax_sparse_attention_split_kv

try:
    from .converter import npu_minimax_sparse_attention_split_kv as _ge_converter  # noqa: F401
except (ImportError, AttributeError):
    _ge_converter = None

__all__ = ["build_k2q_csr", "npu_minimax_sparse_attention_split_kv"] + list(
    module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)])
)

custom_ops_module = getattr(torch.ops, "custom", None)

if custom_ops_module is not None:
    for op_name in dir(custom_ops_module):
        if op_name.startswith("_"):
            continue
        setattr(torch_npu, op_name, getattr(custom_ops_module, op_name))
else:
    WARN_MSG = (
        "torch.ops.custom module is not found, mount custom ops to torch_npu failed."
        "Calling by torch_npu.xxx for custom ops is unsupported, please use torch.ops.custom.xxx."
    )
    warnings.warn(WARN_MSG)
    warnings.filterwarnings("ignore", message=WARN_MSG)
