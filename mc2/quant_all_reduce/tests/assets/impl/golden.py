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

"""Golden for aclnnQuantAllReduce.

QuantAllReduce: dequant(x, scale) -> all_reduce(SUM).
"""

import logging
import numpy as np


def _load_framework():
    """Load golden utils from mc2/common/tests/assets."""
    import importlib.util
    from pathlib import Path

    utils_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "golden_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_golden_utils_{abs(hash(utils_path))}", utils_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.to_torch_f32, mod.fmt_compare_result, mod.apply_goldens_and_compare


def quant_all_reduce_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    to_torch_f32, fmt_compare, apply_compare = _load_framework()

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x = to_torch_f32(tc.flatten_tensors[0])
        scale = tc.flatten_tensors[1] if len(tc.flatten_tensors) > 1 else None
        if scale is not None and isinstance(scale, torch.Tensor) and scale.numel() > 0:
            x = x * to_torch_f32(scale)
        local_results[did] = x

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    rank_goldens = {did: total for did in device_ids}
    apply_compare(thread_contexts, device_ids, rank_goldens, all_precision)


def quant_all_reduce_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    local_results = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        x = inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        if x is None:
            local_results.append(None)
            continue
        scale = inp[1] if len(inp) > 1 and inp[1] is not None else None
        if scale is not None and hasattr(scale, "numel") and scale.numel() > 0:
            x = x * scale.float()
        local_results.append(x)
    total = None
    for r in range(world_size):
        if local_results[r] is not None:
            total = (
                local_results[r].clone() if total is None else total + local_results[r]
            )
    if total is None:
        return [None] * world_size
    return [total.contiguous() for _ in range(world_size)]
