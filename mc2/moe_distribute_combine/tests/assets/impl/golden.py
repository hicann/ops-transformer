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

"""Golden for aclnnMoeDistributeCombine / torch_npu.npu_moe_distribute_combine.

MoeDistributeCombine: all_to_allv(expand_x) -> scatter back to original positions -> sum.
Pure CPU golden, no HCCL cascade (communication simulated on CPU).
Single output: x_out (summed and un-permuted back to original token order).

E2E golden returns a list of x_out tensors (one per rank).
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

    return mod.to_torch_f32, mod.fmt_compare_result, Comparator


def _simulate_moe_alltoallv(all_rank_inputs, world_size, send_counts_per_rank):
    """Simulate all_to_allv on CPU: each rank sends chunks to all other ranks."""
    import torch

    rank_outputs = {}
    for target_rank in range(world_size):
        received_chunks = []
        for src_rank in range(world_size):
            src_data = all_rank_inputs[src_rank]
            src_counts = send_counts_per_rank[src_rank]
            offset = 0
            for dst_idx in range(world_size):
                count = int(src_counts[dst_idx])
                if dst_idx == target_rank and count > 0:
                    received_chunks.append(src_data[offset : offset + count])
                offset += count
        if received_chunks:
            rank_outputs[target_rank] = torch.cat(received_chunks, dim=0)
        else:
            h = all_rank_inputs[0].shape[-1] if all_rank_inputs[0].dim() > 1 else 1
            rank_outputs[target_rank] = torch.zeros(
                0, h, dtype=all_rank_inputs[0].dtype
            )
    return rank_outputs


# ---------------------------------------------------------------------------
# E2E golden: returns list of x_out tensors (one per rank)
# ---------------------------------------------------------------------------


def moe_distribute_combine_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    ep_ws = int(attrs.get("ep_world_size", world_size))
    moe_expert_num = int(attrs.get("moe_expert_num", 1))
    local_expert_num = moe_expert_num // ep_ws if ep_ws > 0 else moe_expert_num

    # Combine inputs per rank:
    #  [0] expand_x: dispatched+received tokens (padded to tp_ws * A_moe)
    #  [1] expert_ids: original expert_ids (bs, k)
    #  [2] expand_idx: indices mapping expand_x rows back to (token, k) positions
    #  [3] ep_send_counts: per-rank send counts (cumsum format, ep_ws values)
    #  [4] expert_scales: per-token scaling factors (bs, k) or (bs*k,)
    all_rank_expand_x = {}
    all_rank_send_counts = {}
    all_rank_expert_ids = {}
    all_rank_expand_idx = {}
    all_rank_expert_scales = {}

    for r in range(world_size):
        inputs = cpu_inputs_per_rank[r]
        expand_x = (
            inputs[0].float() if (len(inputs) > 0 and inputs[0] is not None) else None
        )
        expert_ids = inputs[1] if (len(inputs) > 1 and inputs[1] is not None) else None
        expand_idx = inputs[2] if (len(inputs) > 2 and inputs[2] is not None) else None
        ep_send_counts = (
            inputs[3] if (len(inputs) > 3 and inputs[3] is not None) else None
        )
        expert_scales = (
            inputs[4] if (len(inputs) > 4 and inputs[4] is not None) else None
        )

        if expand_x is None:
            all_rank_expand_x[r] = torch.zeros(0, 1)
            all_rank_send_counts[r] = [0] * ep_ws
            all_rank_expert_ids[r] = torch.zeros(0, 1, dtype=torch.int32)
            all_rank_expand_idx[r] = torch.zeros(0, dtype=torch.int32)
            all_rank_expert_scales[r] = torch.zeros(0)
            continue

        h = expand_x.shape[1]

        # Derive send_counts from ep_send_counts (cumsum format)
        # ep_send_counts is cumsum, so send_counts[i] = ep_send_counts[i] - ep_send_counts[i-1]
        if ep_send_counts is not None and ep_send_counts.numel() > 0:
            esc = ep_send_counts.float()
            send_counts = []
            prev = 0.0
            for i in range(min(ep_ws, esc.numel())):
                cur = float(esc[i])
                send_counts.append(int(cur - prev))
                prev = cur
            while len(send_counts) < ep_ws:
                send_counts.append(0)
        else:
            # Fallback: evenly distribute
            num_tokens = expand_x.shape[0]
            per_rank = num_tokens // ep_ws
            send_counts = [per_rank] * ep_ws
            remainder = num_tokens % ep_ws
            for i in range(remainder):
                send_counts[i] += 1

        all_rank_expand_x[r] = expand_x
        all_rank_send_counts[r] = send_counts
        all_rank_expert_ids[r] = (
            expert_ids
            if expert_ids is not None
            else torch.zeros(0, 1, dtype=torch.int32)
        )
        all_rank_expand_idx[r] = (
            expand_idx if expand_idx is not None else torch.zeros(0, dtype=torch.int32)
        )
        all_rank_expert_scales[r] = (
            expert_scales if expert_scales is not None else torch.zeros(0)
        )

    # Simulate alltoallv: each rank sends its expand_x to destination ranks
    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, world_size, all_rank_send_counts
    )

    # For each rank, scatter the received tokens back to original positions and sum
    goldens = []
    for r in range(world_size):
        a2a_result = alltoallv_out.get(r, torch.zeros(0, 1))
        expert_ids = all_rank_expert_ids[r]
        expand_idx = all_rank_expand_idx[r]
        expert_scales = all_rank_expert_scales[r]
        expand_x = all_rank_expand_x[r]

        h = a2a_result.shape[1] if a2a_result.dim() > 1 else 1
        bs = expert_ids.shape[0] if expert_ids.dim() > 0 else 0
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        x_out = torch.zeros(bs, h)

        if a2a_result.numel() > 0 and bs > 0 and expand_idx.numel() > 0:
            num_valid = min(expand_idx.shape[0], a2a_result.shape[0])
            for idx_pos in range(num_valid):
                orig_flat_idx = int(expand_idx[idx_pos])
                orig_token = orig_flat_idx // k if k > 0 else orig_flat_idx
                k_idx = orig_flat_idx % k if k > 0 else 0
                if orig_token < bs:
                    scale = 1.0
                    if expert_scales.numel() > 0:
                        if (
                            expert_scales.dim() > 1
                            and orig_token < expert_scales.shape[0]
                            and k_idx < expert_scales.shape[1]
                        ):
                            scale = float(expert_scales[orig_token][k_idx])
                        elif (
                            expert_scales.dim() == 1
                            and orig_flat_idx < expert_scales.shape[0]
                        ):
                            scale = float(expert_scales[orig_flat_idx])
                    x_out[orig_token] = (
                        x_out[orig_token] + a2a_result[idx_pos].float() * scale
                    )

        goldens.append(x_out.contiguous())
    return goldens


# ---------------------------------------------------------------------------
# ACLNN multi-device golden
# ---------------------------------------------------------------------------


def moe_distribute_combine_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch

    to_torch_f32, fmt_compare, Comparator = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    world_size = len(device_ids)
    ep_ws = int(attrs.get("epWorldSize", world_size))
    moe_expert_num = int(attrs.get("moeExpertNum", 1))
    local_expert_num = moe_expert_num // ep_ws if ep_ws > 0 else moe_expert_num

    all_rank_expand_x = {}
    all_rank_send_counts = {}
    all_rank_expert_ids = {}
    all_rank_expand_idx = {}
    all_rank_expert_scales = {}

    for did in device_ids:
        tc = thread_contexts[did]
        expand_x = tc.flatten_tensors[0]
        expert_ids = tc.flatten_tensors[1]
        expand_idx = tc.flatten_tensors[2]
        ep_send_counts = tc.flatten_tensors[3]
        expert_scales = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None

        h = expand_x.shape[1]

        if ep_send_counts is not None and ep_send_counts.numel() > 0:
            esc = ep_send_counts.float()
            send_counts = []
            prev = 0.0
            for i in range(min(ep_ws, esc.numel())):
                cur = float(esc[i])
                send_counts.append(int(cur - prev))
                prev = cur
            while len(send_counts) < ep_ws:
                send_counts.append(0)
        else:
            num_tokens = expand_x.shape[0]
            per_rank = num_tokens // ep_ws
            send_counts = [per_rank] * ep_ws
            remainder = num_tokens % ep_ws
            for i in range(remainder):
                send_counts[i] += 1

        all_rank_expand_x[did] = expand_x
        all_rank_send_counts[did] = send_counts
        all_rank_expert_ids[did] = (
            expert_ids
            if expert_ids is not None
            else torch.zeros(0, 1, dtype=torch.int32)
        )
        all_rank_expand_idx[did] = (
            expand_idx if expand_idx is not None else torch.zeros(0, dtype=torch.int32)
        )
        all_rank_expert_scales[did] = (
            expert_scales if expert_scales is not None else torch.zeros(0)
        )

    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, device_ids, all_rank_send_counts
    )

    for did in device_ids:
        tc = thread_contexts[did]
        a2a_result = alltoallv_out[did]
        expert_ids = all_rank_expert_ids[did]
        expand_idx = all_rank_expand_idx[did]
        expert_scales = all_rank_expert_scales[did]

        h = a2a_result.shape[1] if a2a_result.dim() > 1 else 1
        bs = expert_ids.shape[0] if expert_ids.dim() > 0 else 0
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        x_out = torch.zeros(bs, h, dtype=expand_x.dtype)
        if a2a_result.numel() > 0 and bs > 0 and expand_idx.numel() > 0:
            num_valid = min(expand_idx.shape[0], a2a_result.shape[0])
            for idx_pos in range(num_valid):
                orig_flat_idx = int(expand_idx[idx_pos])
                orig_token = orig_flat_idx // k if k > 0 else orig_flat_idx
                k_idx = orig_flat_idx % k if k > 0 else 0
                if orig_token < bs:
                    scale = 1.0
                    if expert_scales is not None and expert_scales.numel() > 0:
                        if (
                            expert_scales.dim() > 1
                            and orig_token < expert_scales.shape[0]
                            and k_idx < expert_scales.shape[1]
                        ):
                            scale = float(expert_scales[orig_token][k_idx])
                        elif (
                            expert_scales.dim() == 1
                            and orig_flat_idx < expert_scales.shape[0]
                        ):
                            scale = float(expert_scales[orig_flat_idx])
                    x_out[orig_token] = (
                        x_out[orig_token] + a2a_result[idx_pos].float() * scale
                    )

        x_out = x_out.to(expand_x.dtype)
        tc.golden_tensors = [x_out.contiguous()]
        try:
            cr = Comparator(tc).compare()
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"MoeDistributeCombine: rank dev={did} FAILED: {cr.precision}"
                )
            else:
                logging.info(f"MoeDistributeCombine: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"MoeDistributeCombine: rank dev={did} compare failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")
