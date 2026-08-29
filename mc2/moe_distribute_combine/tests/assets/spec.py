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

"""TestSpec adapter for MoeDistributeCombine TTK assets.

All impl modules are loaded lazily to avoid import-time failures.
"""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def _load_impl_module(stem):
    if stem not in _impl_cache:
        path = ASSET_IMPL_DIR / f"{stem}.py"
        spec = importlib.util.spec_from_file_location(
            f"moe_distribute_combine_assets_impl_{stem}_{abs(hash(path))}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _impl_cache[stem] = module
    return _impl_cache[stem]


class AclnnMoeDistributeCombineSpec:
    """TestSpec for aclnnMoeDistributeCombine (ACLNN multi-device path)."""

    @staticmethod
    def golden(thread_contexts, device_ids, all_precision):
        return _load_impl_module("golden").moe_distribute_combine_multi_device_golden(
            thread_contexts, device_ids, all_precision
        )


class MoeDistributeCombineE2ESpec:
    """TestSpec for torch_npu.npu_moe_distribute_combine (E2E multi-device path)."""

    @staticmethod
    def golden(cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True):
        return _load_impl_module("golden").moe_distribute_combine_e2e_golden(
            cpu_inputs_per_rank, attrs, rank, ws, dist_avail
        )

    @staticmethod
    def input(
        expand_x,
        expert_ids,
        expand_idx,
        ep_send_counts,
        expert_scales,
        group_ep,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        *,
        tp_send_counts=None,
        x_active_mask=None,
        activation_scale=None,
        weight_scale=None,
        group_list=None,
        expand_scales=None,
        shared_expert_x=None,
        group_tp="",
        tp_world_size=0,
        tp_rank_id=0,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        global_bs=0,
        out_dtype=0,
        comm_quant_mode=0,
        group_list_type=0,
        **kwargs,
    ):
        """Generate consistent dispatch-then-combine inputs.

        The combine operator needs valid ep_send_counts that come from an actual
        dispatch operation. This input plugin simulates dispatch to generate
        consistent expand_x, expand_idx, and ep_send_counts from expert_ids.

        Since the E2E multi-device path generates inputs per-rank in the parent
        process, we use the testcase_name to derive a rank from the input data
        range seed. The actual per-rank ep_rank_id override happens in the worker.
        """
        import torch
        import numpy as np

        eid_np = (
            expert_ids.numpy()
            if hasattr(expert_ids, "numpy")
            else np.asarray(expert_ids)
        )
        bs = eid_np.shape[0]
        k = eid_np.shape[1] if eid_np.ndim > 1 else 1
        h = expand_x.shape[1] if hasattr(expand_x, "shape") else 7168
        local_expert_num = (
            moe_expert_num // ep_world_size if ep_world_size > 0 else moe_expert_num
        )

        # Simulate dispatch: sort tokens by destination rank, compute send_counts
        send_counts = [0] * ep_world_size
        token_groups = [[] for _ in range(ep_world_size)]
        for i in range(bs):
            for j in range(k):
                eid = int(eid_np[i][j]) if eid_np.ndim > 1 else int(eid_np[i])
                dest = eid // local_expert_num if local_expert_num > 0 else 0
                if dest >= ep_world_size:
                    dest = ep_world_size - 1
                send_counts[dest] += 1
                # Store the x row for this token
                if hasattr(expand_x, "numpy"):
                    token_groups[dest].append(expand_x.numpy()[i])

        # Build expand_x: sorted by destination rank, then padded to A_moe
        sorted_x = []
        for dest in range(ep_world_size):
            if token_groups[dest]:
                sorted_x.extend(token_groups[dest])
        total_actual = sum(send_counts)

        a_moe = bs * ep_world_size * min(local_expert_num, k)
        target_len = a_moe  # tp_ws=1

        if sorted_x:
            expand_x_data = np.stack(sorted_x, axis=0)
        else:
            expand_x_data = np.zeros((0, h), dtype=np.float32)

        # Pad to target_len
        if expand_x_data.shape[0] < target_len:
            pad = np.zeros(
                (target_len - expand_x_data.shape[0], h), dtype=expand_x_data.dtype
            )
            expand_x_data = np.concatenate([expand_x_data, pad], axis=0)
        elif expand_x_data.shape[0] > target_len:
            expand_x_data = expand_x_data[:target_len]

        # ep_send_counts: cumsum format
        cumsum = []
        running = 0
        for i in range(ep_world_size):
            running += send_counts[i]
            cumsum.append(running)
        ep_send_counts_np = np.array(cumsum, dtype=np.int32)

        # expand_idx: flat indices (token * k + k_idx) for each sorted token
        expand_idx_list = []
        for i in range(bs):
            for j in range(k):
                expand_idx_list.append(i * k + j)
        expand_idx_np = np.array(expand_idx_list, dtype=np.int32)

        # expert_scales: set to 1.0
        expert_scales_np = np.ones((bs, k), dtype=np.float32)

        # Write back in-place
        expand_x[:] = torch.from_numpy(
            expand_x_data.astype(
                np.float16 if expand_x.dtype == torch.bfloat16 else expand_x_data.dtype
            )
        )
        ep_send_counts[:] = torch.from_numpy(ep_send_counts_np)
        expand_idx[:] = torch.from_numpy(expand_idx_np)
        expert_scales[:] = torch.from_numpy(expert_scales_np)


__spec__ = {
    "aclnnMoeDistributeCombine": "AclnnMoeDistributeCombineSpec",
    "torch_npu.npu_moe_distribute_combine": "MoeDistributeCombineE2ESpec",
    "cann_ops_transformer.npu_moe_distribute_combine": "MoeDistributeCombineE2ESpec",
}
