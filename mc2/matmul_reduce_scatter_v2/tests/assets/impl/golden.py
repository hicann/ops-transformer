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

"""Golden for aclnnMatmulReduceScatterV2.

MatmulReduceScatterV2 extends V1 with quant support.
Delegates to the shared reduce_scatter golden logic.
"""

import importlib.util
from pathlib import Path


def _load_shared_utils():
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matmul_reduce_scatter_v2_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    """Use shared golden_reduce_scatter_compare."""
    utils = _load_shared_utils()
    world_size = len(device_ids)
    utils.golden_reduce_scatter_compare(
        thread_contexts, device_ids, all_precision, world_size
    )


def matmul_reduce_scatter_v2_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    local_results = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        x1 = inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        x2 = inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        if x1 is None or x2 is None:
            local_results.append(None)
            continue
        mm_out = torch.matmul(x1, x2)
        bias = inp[2] if len(inp) > 2 and inp[2] is not None else None
        if bias is not None and hasattr(bias, "numel") and bias.numel() > 0:
            mm_out = mm_out + bias.float()
        local_results.append(mm_out)
    total = None
    for r in range(world_size):
        if local_results[r] is not None:
            if total is None:
                total = local_results[r].clone()
            else:
                total = total + local_results[r]
    if total is None:
        return [None] * world_size
    M = total.shape[0]
    chunk_m = M // world_size
    goldens = []
    for idx in range(world_size):
        goldens.append(total[idx * chunk_m : (idx + 1) * chunk_m, :].contiguous())
    return goldens
