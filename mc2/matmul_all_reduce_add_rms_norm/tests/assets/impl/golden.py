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

"""Golden for aclnnMatmulAllReduceAddRmsNorm.

MatmulAllReduceAddRmsNorm: matmul(x1,x2)+bias -> all_reduce(SUM) -> add_rms_norm(residual, gamma).
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


def matmul_all_reduce_add_rms_norm_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch

    to_torch_f32, fmt_compare, apply_compare = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    eps = float(attrs.get("epsilon", 1e-6))

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = to_torch_f32(tc.flatten_tensors[0])
        x2 = to_torch_f32(tc.flatten_tensors[1])
        mm_out = torch.matmul(x1, x2)
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() > 0:
            mm_out = mm_out + to_torch_f32(bias)
        local_results[did] = mm_out

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    # Add RMS Norm: out = all_reduce + residual, then rms_norm
    rank_goldens = {}
    for did in device_ids:
        tc = thread_contexts[did]
        residual = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
        gamma = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        out = total.clone()
        if (
            residual is not None
            and isinstance(residual, torch.Tensor)
            and residual.numel() > 0
        ):
            out = out + to_torch_f32(residual)
        if gamma is not None and isinstance(gamma, torch.Tensor) and gamma.numel() > 0:
            gamma_f = to_torch_f32(gamma)
            rms = torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
            out = out / rms * gamma_f
        rank_goldens[did] = out
    apply_compare(thread_contexts, device_ids, rank_goldens, all_precision)


def matmul_all_reduce_add_rms_norm_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    eps = float(attrs.get("epsilon", 1e-6))
    local_results = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        x1 = inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        x2 = inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        if x1 is None or x2 is None:
            local_results.append(None)
            continue
        mm_out = torch.matmul(x1, x2)
        local_results.append(mm_out)
    total = None
    for r in range(world_size):
        if local_results[r] is not None:
            total = (
                local_results[r].clone() if total is None else total + local_results[r]
            )
    if total is None:
        return [None] * world_size
    goldens = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        out = total.clone()
        residual = inp[3] if len(inp) > 3 and inp[3] is not None else None
        gamma = inp[4] if len(inp) > 4 and inp[4] is not None else None
        if residual is not None and hasattr(residual, "numel") and residual.numel() > 0:
            out = out + residual.float()
        if gamma is not None and hasattr(gamma, "numel") and gamma.numel() > 0:
            gamma_f = gamma.float()
            rms = torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
            out = out / rms * gamma_f
        goldens.append(out.contiguous())
    return goldens
