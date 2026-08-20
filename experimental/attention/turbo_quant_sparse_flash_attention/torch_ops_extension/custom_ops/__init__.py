# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import torch
import torch_npu

from . import custom_ops_lib  # noqa: F401

_OP_NAME = "npu_turbo_quant_sparse_flash_attention"
setattr(torch_npu, _OP_NAME, getattr(torch.ops.custom, _OP_NAME))

__all__ = [_OP_NAME]
