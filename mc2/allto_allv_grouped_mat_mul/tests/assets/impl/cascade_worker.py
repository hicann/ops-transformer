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

"""Standalone cascade worker for AlltoAllvGroupedMatMul.

Flow: all_to_allv(gmm_x) -> permute -> npu_gmm -> output [+ permute_out] [+ mm_out]
Args: rank world_size port input_path result_path error_path
      trans_gmm_weight trans_mm_weight permute_out_flag mm_out_flag
"""

import os
import sys
import datetime
import traceback
from itertools import accumulate

import numpy as np
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from mindspeed.ops import gmm


def _load_inputs_for_rank(input_path, did):
    data = np.load(input_path, allow_pickle=False)
    tensors = {}
    for key in data.files:
        if key.startswith(f"did{did}_t") and not key.endswith("_dtype"):
            idx = int(key.split("_t")[1])
            arr = data[key]
            dtype_key = f"{key}_dtype"
            dtype_str = str(data[dtype_key]) if dtype_key in data.files else "float32"
            if arr.dtype.kind == "V":
                total_bytes = arr.size * arr.dtype.itemsize
                arr = np.frombuffer(arr.tobytes(), dtype=np.uint8, count=total_bytes)
                arr = arr.reshape(arr.shape if arr.ndim > 0 else (1,))
            t = torch.from_numpy(arr.copy())
            if dtype_str == "torch.bfloat16":
                t = t.to(torch.bfloat16)
            elif dtype_str == "torch.float16":
                t = t.to(torch.float16)
            tensors[idx] = t
    return tensors


def _permute_a2a_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    device = tokens.device
    indices = torch.zeros(exp_per_card, ep_ws).long().to(device)
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = expTokenNums[i][j + exp_per_card * rank_idx]
    trans = indices.permute(1, 0)
    flaten = trans.reshape(-1)
    sum_list = torch.cumsum(flaten, dim=0)
    tmp = []
    for i in range(len(sum_list)):
        if i == 0:
            tmp.append(range(0, int(sum_list[i])))
        else:
            tmp.append(range(int(sum_list[i - 1]), int(sum_list[i])))
    out = []
    expert_sizes = []
    for e in range(exp_per_card):
        exp_token = []
        for r in range(ep_ws):
            exp_token += list(tmp[e + r * exp_per_card])
        combined = torch.tensor(exp_token).long().to(device)
        out.append(tokens.index_select(0, combined))
        expert_sizes.append(len(exp_token))
    return torch.cat(out, dim=0), expert_sizes


def _gmm_group_list_cumsum(expTokenNums, rank_idx, exp_per_card, ep_ws):
    group_list = []
    for j in range(exp_per_card):
        total = sum(expTokenNums[i][rank_idx * exp_per_card + j] for i in range(ep_ws))
        group_list.append(total)
    return list(accumulate(group_list))


def main():
    (
        rank,
        world_size,
        port,
        input_path,
        result_path,
        error_path,
        trans_gmm_weight,
        trans_mm_weight,
        permute_out_flag,
        mm_out_flag,
    ) = sys.argv[1:11]
    rank = int(rank)
    world_size = int(world_size)
    trans_gmm_weight = trans_gmm_weight == "True"
    trans_mm_weight = trans_mm_weight == "True"
    permute_out_flag = permute_out_flag == "True"
    mm_out_flag = mm_out_flag == "True"
    try:
        os.environ["HCCL_EXEC_TIMEOUT"] = "3600"
        os.environ["HCCL_LINK_TIMEOUT"] = "3600"
        os.environ["HCCL_CONNECT_TIMEOUT"] = "3600"
        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=3600),
        )
        meta = np.load(input_path + ".meta.npz", allow_pickle=False)
        expTokenNums = meta["expTokenNums"].tolist()
        ep_ws = int(meta["ep_ws"][0])
        exp_per_card = int(meta["exp_per_card"][0])
        tensors = _load_inputs_for_rank(input_path, rank)
        gmm_x = tensors[0].npu()
        gmm_weight = tensors[1].npu()
        mm_x = (
            tensors[4].npu()
            if 4 in tensors and tensors[4] is not None and tensors[4].numel() > 0
            else None
        )
        mm_weight = (
            tensors[5].npu()
            if 5 in tensors and tensors[5] is not None and tensors[5].numel() > 0
            else None
        )
        if trans_gmm_weight:
            gmm_weight = torch.transpose(gmm_weight, 1, 2).contiguous()
        if trans_mm_weight and mm_out_flag and mm_weight is not None:
            mm_weight = torch.transpose(mm_weight, 0, 1).contiguous()
        my_row = expTokenNums[rank]
        input_splits = [
            int(sum(my_row[t * exp_per_card : (t + 1) * exp_per_card]))
            for t in range(ep_ws)
        ]
        output_splits = [
            int(sum(expTokenNums[i][rank * exp_per_card : (rank + 1) * exp_per_card]))
            for i in range(ep_ws)
        ]
        K = gmm_x.shape[1]
        a2a_out = torch.empty(
            sum(output_splits), K, dtype=gmm_x.dtype, device=f"npu:{rank}"
        )
        dist.all_to_all_single(
            a2a_out,
            gmm_x,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )
        torch.npu.synchronize()
        permuted, expert_sizes = _permute_a2a_gmm(
            a2a_out, exp_per_card, ep_ws, rank, expTokenNums
        )
        permuted = permuted.npu()
        group_list = _gmm_group_list_cumsum(expTokenNums, rank, exp_per_card, ep_ws)
        group_list_tensor = torch.tensor(group_list, dtype=torch.int64).npu()
        gmm_out = gmm.npu_gmm(
            permuted, gmm_weight, bias=None, group_list=group_list_tensor, group_type=0
        )
        torch.npu.synchronize()
        gmm_cpu = gmm_out.reshape(-1).contiguous().cpu()
        np.savez(
            f"{result_path}.did{rank}.npz", **{f"cascade_did{rank}": gmm_cpu.numpy()}
        )
        if permute_out_flag:
            permute_cpu = permuted.reshape(-1).contiguous().cpu()
            np.savez(
                f"{result_path}.a2a_did{rank}.npz",
                **{f"cascade_a2a_did{rank}": permute_cpu.numpy()},
            )
        if mm_out_flag and mm_x is not None:
            mm_out = torch.mm(mm_x, mm_weight)
            mm_cpu = mm_out.reshape(-1).contiguous().cpu()
            np.savez(
                f"{result_path}.mm_did{rank}.npz",
                **{f"cascade_mm_did{rank}": mm_cpu.numpy()},
            )
        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
