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

"""Golden for aclnnMatmulAllReduce / torch_npu.npu_mm_all_reduce_base.

MatmulAllReduce: matmul(x1, x2) + bias -> all_reduce(SUM).
Handles V2 (extra x3 add), WeightQuant, and QuantMatmul variants.
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
    from ttk.core_modules.npu.op_api.comparison import Comparator

    return (
        mod.to_torch_f32,
        mod.fmt_compare_result,
        mod.apply_goldens_and_compare,
        mod.get_transpose_flags,
        Comparator,
    )


# ---------------------------------------------------------------------------
# ACLNN multi-device golden
# ---------------------------------------------------------------------------


def matmul_all_reduce_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    to_torch_f32, fmt_compare, apply_compare, get_transpose_flags, Comparator = (
        _load_framework()
    )
    first_ctx = next(iter(thread_contexts.values()))
    api_name = first_ctx.api_name
    attrs = first_ctx.attributes

    is_weight_quant = "WeightQuantMatmulAllReduce" in api_name
    is_quant_matmul = "QuantMatmulAllReduce" in api_name and "Weight" not in api_name
    is_v2 = "AllReduceV2" in api_name or "AllReduceV3" in api_name

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]

        if is_weight_quant:
            x1_f = to_torch_f32(x1)
            x2_f = to_torch_f32(x2)
            aq_scale = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            aq_offset = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
            aq_scale_f = to_torch_f32(aq_scale) if aq_scale is not None else None
            aq_offset_f = (
                to_torch_f32(aq_offset)
                if aq_offset is not None
                and isinstance(aq_offset, torch.Tensor)
                and aq_offset.numel() > 0
                else None
            )
            group_size = int(attrs.get("antiquantGroupSize", 0))
            if group_size > 0 and aq_scale_f is not None:
                aq_scale_f = aq_scale_f.repeat_interleave(group_size, dim=0)
                if aq_offset_f is not None:
                    aq_offset_f = aq_offset_f.repeat_interleave(group_size, dim=0)
            if aq_offset_f is not None:
                weight_deq = (x2_f + aq_offset_f) * aq_scale_f
            elif aq_scale_f is not None:
                weight_deq = x2_f * aq_scale_f
            else:
                weight_deq = x2_f
            mm_out = torch.matmul(x1_f, weight_deq)
        elif is_quant_matmul:
            is_v4_v5 = (
                "QuantMatmulAllReduceV4" in api_name
                or "QuantMatmulAllReduceV5" in api_name
            )
            if is_v4_v5:
                x1_f = to_torch_f32(x1)
                x2_f = to_torch_f32(x2)
                mm_out = torch.matmul(x1_f, x2_f)
                x1scale = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                x2scale = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
                if x1scale is not None:
                    x1s_f = to_torch_f32(x1scale)
                    if x1s_f.dim() == 1 and mm_out.dim() == 2:
                        x1s_f = x1s_f.unsqueeze(-1)
                    elif x1s_f.dim() == 1 and mm_out.dim() == 3:
                        x1s_f = x1s_f.unsqueeze(0).unsqueeze(-1)
                    mm_out = mm_out * x1s_f
                if x2scale is not None:
                    x2s_f = to_torch_f32(x2scale)
                    mm_out = mm_out * x2s_f
                ds = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
                if ds is not None:
                    ds_f = to_torch_f32(ds)
                    if ds_f.dim() == 1 and mm_out.dim() >= 2:
                        ds_f = ds_f.unsqueeze(0)
                    mm_out = mm_out * ds_f
                x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                if x3 is not None and hasattr(x3, "numel") and x3.numel() > 0:
                    mm_out = mm_out + to_torch_f32(x3)
            else:
                x1_f = to_torch_f32(x1)
                x2_f = to_torch_f32(x2)
                mm_out = torch.matmul(x1_f, x2_f)
                is_v2_quant = "QuantMatmulAllReduceV2" in api_name
                if is_v2_quant:
                    ds = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                    if ds is not None:
                        mm_out = mm_out * to_torch_f32(ds)
                    pt = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
                    if (
                        pt is not None
                        and isinstance(pt, torch.Tensor)
                        and pt.numel() > 0
                    ):
                        pt_f = to_torch_f32(pt)
                        if pt_f.dim() == 1:
                            pt_f = pt_f.unsqueeze(1)
                        mm_out = mm_out * pt_f
                    x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                    if (
                        x3 is not None
                        and isinstance(x3, torch.Tensor)
                        and x3.numel() > 0
                    ):
                        mm_out = mm_out + to_torch_f32(x3)
                else:
                    ds = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
                    if ds is not None:
                        mm_out = mm_out * to_torch_f32(ds)
                    bias = (
                        tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                    )
                    if (
                        bias is not None
                        and isinstance(bias, torch.Tensor)
                        and bias.numel() == 0
                    ):
                        bias = None
                    if bias is not None:
                        mm_out = mm_out + to_torch_f32(bias)
                    if "V3" in api_name and len(tc.flatten_tensors) > 5:
                        pt = tc.flatten_tensors[5]
                        if (
                            pt is not None
                            and isinstance(pt, torch.Tensor)
                            and pt.numel() > 0
                        ):
                            pt_f = to_torch_f32(pt)
                            if pt_f.dim() == 1:
                                pt_f = pt_f.unsqueeze(1)
                            mm_out = mm_out * pt_f
        else:
            bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            if (
                bias is not None
                and isinstance(bias, torch.Tensor)
                and bias.numel() == 0
            ):
                bias = None
            x1_f = to_torch_f32(x1)
            x2_f = to_torch_f32(x2)
            mm_out = torch.matmul(x1_f, x2_f)
            if bias is not None:
                mm_out = mm_out + to_torch_f32(bias)
            if is_v2:
                x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                if x3 is not None and isinstance(x3, torch.Tensor) and x3.numel() > 0:
                    mm_out = mm_out + to_torch_f32(x3)

        x1_dtype = x1.dtype if hasattr(x1, "dtype") else None
        if x1_dtype is not None and x1_dtype in (torch.bfloat16, torch.float16):
            mm_out = mm_out.to(x1_dtype).float()
        local_results[did] = mm_out

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    rank_goldens = {did: total for did in device_ids}
    apply_compare(thread_contexts, device_ids, rank_goldens, all_precision)


# ---------------------------------------------------------------------------
# E2E golden
# ---------------------------------------------------------------------------


def matmul_all_reduce_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    all_x1 = [
        inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        for inp in cpu_inputs_per_rank
    ]
    all_x2 = [
        inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        for inp in cpu_inputs_per_rank
    ]

    local_results = []
    for r in range(world_size):
        x1 = all_x1[r]
        x2 = all_x2[r]
        if x1 is None or x2 is None:
            local_results.append(None)
            continue
        mm_out = torch.matmul(x1, x2)
        inputs = cpu_inputs_per_rank[r]
        bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
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
    return [total.contiguous() for _ in range(world_size)]
