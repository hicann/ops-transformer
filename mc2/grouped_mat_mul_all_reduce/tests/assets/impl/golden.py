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

"""Golden for aclnnGroupedMatMulAllReduce.

GroupedMatMul: matmul per group -> all_reduce(SUM).
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
    return (
        mod.to_torch_f32,
        mod.fmt_compare_result,
        mod.apply_goldens_and_compare,
        mod.grouped_matmul_cpu,
    )


def grouped_mat_mul_all_reduce_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch

    to_torch_f32, fmt_compare, apply_compare, gmm_cpu = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        gmm_x = to_torch_f32(tc.flatten_tensors[0])
        gmm_w = to_torch_f32(tc.flatten_tensors[1])
        group_list = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if group_list is not None and isinstance(group_list, torch.Tensor):
            group_list = group_list.to(torch.int64)
        gmm_out = gmm_cpu(gmm_x, gmm_w, group_list)
        local_results[did] = gmm_out

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    rank_goldens = {did: total for did in device_ids}
    apply_compare(thread_contexts, device_ids, rank_goldens, all_precision)


def grouped_mat_mul_all_reduce_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    local_results = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        gmm_x = inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        gmm_w = inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        if gmm_x is None or gmm_w is None:
            local_results.append(None)
            continue
        group_list = inp[2] if len(inp) > 2 and inp[2] is not None else None
        if group_list is not None:
            group_list = group_list.to(torch.int64)
        # Simple grouped matmul: if group_list is None, do regular matmul
        if group_list is None:
            local_results.append(torch.matmul(gmm_x, gmm_w))
        else:
            # Split by group_list and matmul each segment
            parts = []
            start = 0
            for g in group_list.flatten().tolist():
                end = int(g)
                parts.append(torch.matmul(gmm_x[start:end], gmm_w))
                start = end
            local_results.append(torch.cat(parts, dim=0))
    total = None
    for r in range(world_size):
        if local_results[r] is not None:
            total = (
                local_results[r].clone() if total is None else total + local_results[r]
            )
    if total is None:
        return [None] * world_size
    return [total.contiguous() for _ in range(world_size)]
