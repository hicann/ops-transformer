#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden for aclnnInplaceMatmulAllReduceAddRmsNorm.

Same computation as MatmulAllReduceAddRmsNorm but with inplace output.
"""

import logging
import numpy as np

# Reuse the same golden logic
from pathlib import Path
import importlib.util

_impl_dir = (
    Path(__file__).resolve().parent.parent.parent
    / "matmul_all_reduce_add_rms_norm"
    / "tests"
    / "assets"
    / "impl"
)
_spec = importlib.util.spec_from_file_location(
    "_mmaarn_golden", _impl_dir / "golden.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def inplace_matmul_all_reduce_add_rms_norm_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    return _mod.matmul_all_reduce_add_rms_norm_multi_device_golden(
        thread_contexts, device_ids, all_precision
    )


def inplace_matmul_all_reduce_add_rms_norm_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    return _mod.matmul_all_reduce_add_rms_norm_e2e_golden(
        cpu_inputs_per_rank, attrs, rank, ws, dist_avail
    )
