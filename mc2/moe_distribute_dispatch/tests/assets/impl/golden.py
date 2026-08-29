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

"""Golden for aclnnMoeDistributeDispatch / torch_npu.npu_moe_distribute_dispatch.

MoeDistributeDispatch: sort tokens by destination expert -> all_to_allv -> output.
Pure CPU golden, no HCCL cascade (communication simulated on CPU).
Multi-output: recv_x, dynamic_scales, expand_idx, expert_token_nums,
              ep_recv_counts, tp_recv_counts, expand_scales.

E2E golden returns a list (one per rank) of the recv_x output tensor.
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
# E2E golden: returns list of recv_x tensors (one per rank)
# ---------------------------------------------------------------------------


def moe_distribute_dispatch_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    ep_ws = int(attrs.get("ep_world_size", world_size))
    moe_expert_num = int(attrs.get("moe_expert_num", 1))
    local_expert_num = moe_expert_num // ep_ws if ep_ws > 0 else moe_expert_num

    all_rank_expand_x = {}
    all_rank_send_counts = {}

    for r in range(world_size):
        inputs = cpu_inputs_per_rank[r]
        x = inputs[0].float() if inputs[0] is not None else None
        expert_ids = inputs[1] if (len(inputs) > 1 and inputs[1] is not None) else None
        if x is None or expert_ids is None:
            all_rank_expand_x[r] = torch.zeros(0, 1)
            all_rank_send_counts[r] = [0] * ep_ws
            continue

        bs = x.shape[0]
        h = x.shape[1]
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        # Sort tokens by destination rank (expert_id // local_expert_num)
        send_counts = [0] * ep_ws
        token_groups = [[] for _ in range(ep_ws)]
        for i in range(bs):
            for j in range(k):
                eid = (
                    int(expert_ids[i][j])
                    if expert_ids.dim() > 1
                    else int(expert_ids[i])
                )
                dest_rank = eid // local_expert_num if local_expert_num > 0 else 0
                if dest_rank >= ep_ws:
                    dest_rank = ep_ws - 1
                token_groups[dest_rank].append(x[i])
                send_counts[dest_rank] += 1

        sorted_tokens = []
        for dest in range(ep_ws):
            sorted_tokens.extend(token_groups[dest])

        if sorted_tokens:
            expand_x_local = torch.stack(sorted_tokens, dim=0)
        else:
            expand_x_local = torch.zeros(0, h)

        all_rank_expand_x[r] = expand_x_local
        all_rank_send_counts[r] = send_counts

    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, world_size, all_rank_send_counts
    )

    # The NPU expand_x output is padded to tp_ws * A_moe per rank, where
    # A_moe = global_bs * min(local_expert_num, k) = bs * ep_ws * min(local_expert_num, k).
    # We pad the golden recv_x to match this size for isclose comparison.
    first_x = cpu_inputs_per_rank[0][0]
    bs = first_x.shape[0] if first_x is not None else 0
    h = first_x.shape[1] if first_x is not None else 1
    k = 1
    for inp in cpu_inputs_per_rank:
        if len(inp) > 1 and inp[1] is not None:
            eid = inp[1]
            k = eid.shape[1] if eid.dim() > 1 else 1
            break
    tp_ws = int(attrs.get("tp_world_size", 1))
    a_moe = bs * ep_ws * min(local_expert_num, k)
    target_len = tp_ws * a_moe

    goldens = []
    for r in range(world_size):
        recv_x = alltoallv_out.get(r, torch.zeros(0, h))
        if recv_x.shape[0] < target_len:
            pad_size = target_len - recv_x.shape[0]
            pad = torch.zeros(pad_size, h, dtype=recv_x.dtype)
            recv_x = torch.cat([recv_x, pad], dim=0)
        elif recv_x.shape[0] > target_len:
            recv_x = recv_x[:target_len]
        goldens.append(recv_x.contiguous())
    return goldens


# ---------------------------------------------------------------------------
# ACLNN multi-device golden
# ---------------------------------------------------------------------------


def moe_distribute_dispatch_multi_device_golden(
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
    all_rank_expand_idx = {}
    all_rank_send_counts = {}
    all_rank_expert_token_nums = {}
    all_rank_dynamic_scales = {}
    all_rank_expand_scales = {}

    for did in device_ids:
        tc = thread_contexts[did]
        x = tc.flatten_tensors[0]
        expert_ids = tc.flatten_tensors[1]
        scales_tensor = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        expert_scales = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None

        bs = x.shape[0]
        h = x.shape[1]
        k = expert_ids.shape[1] if expert_ids.dim() > 1 else 1

        send_counts = [0] * ep_ws
        token_groups = [[] for _ in range(ep_ws)]
        expand_idx_list = []
        dynamic_scales_list = []
        expand_scales_list = []

        for i in range(bs):
            for j in range(k):
                eid = (
                    int(expert_ids[i][j])
                    if expert_ids.dim() > 1
                    else int(expert_ids[i])
                )
                dest_rank = eid // local_expert_num if local_expert_num > 0 else 0
                if dest_rank >= ep_ws:
                    dest_rank = ep_ws - 1
                token_groups[dest_rank].append(x[i])
                expand_idx_list.append(i * k + j)
                send_counts[dest_rank] += 1
                if scales_tensor is not None and scales_tensor.numel() > 0:
                    s = (
                        float(scales_tensor[i][j])
                        if scales_tensor.dim() > 1
                        else float(scales_tensor[i])
                    )
                    dynamic_scales_list.append(s)
                if expert_scales is not None and expert_scales.numel() > 0:
                    es = (
                        float(expert_scales[i][j])
                        if expert_scales.dim() > 1
                        else float(expert_scales[i])
                    )
                    expand_scales_list.append(es)

        sorted_tokens = []
        for dest in range(ep_ws):
            sorted_tokens.extend(token_groups[dest])

        if sorted_tokens:
            expand_x_local = torch.stack(sorted_tokens, dim=0).to(x.dtype)
        else:
            expand_x_local = torch.zeros(0, h, dtype=x.dtype)

        all_rank_expand_x[did] = expand_x_local
        all_rank_expand_idx[did] = torch.tensor(expand_idx_list, dtype=torch.int32)
        all_rank_send_counts[did] = send_counts
        all_rank_dynamic_scales[did] = (
            torch.tensor(dynamic_scales_list, dtype=torch.float32)
            if dynamic_scales_list
            else torch.zeros(0, dtype=torch.float32)
        )
        all_rank_expand_scales[did] = (
            torch.tensor(expand_scales_list, dtype=torch.float32)
            if expand_scales_list
            else torch.zeros(0, dtype=torch.float32)
        )

    for did in device_ids:
        rank_idx = list(device_ids).index(did)
        expert_token_nums = [0] * local_expert_num
        for src_did in device_ids:
            src_tc = thread_contexts[src_did]
            src_expert_ids = src_tc.flatten_tensors[1]
            src_bs = src_expert_ids.shape[0]
            src_k = src_expert_ids.shape[1] if src_expert_ids.dim() > 1 else 1
            for i in range(src_bs):
                for j in range(src_k):
                    eid = (
                        int(src_expert_ids[i][j])
                        if src_expert_ids.dim() > 1
                        else int(src_expert_ids[i])
                    )
                    dest_rank = eid // local_expert_num if local_expert_num > 0 else 0
                    if dest_rank >= ep_ws:
                        dest_rank = ep_ws - 1
                    if dest_rank == rank_idx:
                        local_eid = (
                            eid % local_expert_num if local_expert_num > 0 else 0
                        )
                        expert_token_nums[local_eid] += 1
        all_rank_expert_token_nums[did] = torch.tensor(
            expert_token_nums, dtype=torch.int64
        )

    alltoallv_out = _simulate_moe_alltoallv(
        all_rank_expand_x, device_ids, all_rank_send_counts
    )

    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        recv_x = alltoallv_out[did]

        ep_recv_counts = []
        for src_did in device_ids:
            src_counts = all_rank_send_counts[src_did]
            ep_recv_counts.append(src_counts[rank_idx])
        ep_recv_counts_tensor = torch.tensor(ep_recv_counts, dtype=torch.int32)

        tp_ws = int(attrs.get("tpWorldSize", 0))
        tp_recv_counts_tensor = torch.zeros(max(tp_ws, 1), dtype=torch.int32)

        goldens = [
            recv_x.contiguous(),
            all_rank_dynamic_scales[did].contiguous(),
            all_rank_expand_idx[did].contiguous(),
            all_rank_expert_token_nums[did].contiguous(),
            ep_recv_counts_tensor.contiguous(),
            tp_recv_counts_tensor.contiguous(),
            all_rank_expand_scales[did].contiguous(),
        ]
        tc.golden_tensors = goldens
        try:
            cr = Comparator(tc).compare()
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"MoeDistributeDispatch: rank dev={did} FAILED: {cr.precision}"
                )
            else:
                logging.info(f"MoeDistributeDispatch: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"MoeDistributeDispatch: rank dev={did} compare failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")
