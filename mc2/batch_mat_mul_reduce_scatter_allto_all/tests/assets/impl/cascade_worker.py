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

"""Standalone cascade worker for BatchMatMulReduceScatterAlltoAll.

Flow: bmm -> reduce_scatter(TP) -> all_to_all(EP) -> output
Dual comm domain: EP groups do all_to_all, TP groups do reduce_scatter.
Args: rank world_size port input_path result_path error_path
      ep_ws tp_ws shard_type is_trans is_bias
"""

import os
import sys
import datetime
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from torch.distributed import ReduceOp


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


def main():
    (
        rank,
        world_size,
        port,
        input_path,
        result_path,
        error_path,
        ep_ws,
        tp_ws,
        shard_type,
        is_trans,
        is_bias,
    ) = sys.argv[1:12]
    rank = int(rank)
    world_size = int(world_size)
    ep_ws = int(ep_ws)
    tp_ws = int(tp_ws)
    shard_type = int(shard_type)
    is_trans = is_trans == "True"
    is_bias = is_bias == "True"
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
        ep_group = None
        tp_group = None
        # EP groups: strided by tp_ws (matching TTK profiling.py HCCL comm creation)
        for i in range(tp_ws):
            ep_ranks = [x * tp_ws + i for x in range(ep_ws)]
            g = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = g
        # TP groups: contiguous (matching TTK profiling.py HCCL comm creation)
        for i in range(ep_ws):
            tp_ranks = [x + tp_ws * i for x in range(tp_ws)]
            g = dist.new_group(backend="hccl", ranks=tp_ranks)
            if rank in tp_ranks:
                tp_group = g

        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = None
        if (
            is_bias
            and 2 in tensors
            and tensors[2] is not None
            and tensors[2].numel() > 0
        ):
            bias = tensors[2].npu()
        weight_mat = x2

        bmm_out = torch.bmm(x1, weight_mat)
        E_div_ep = x1.shape[0]
        x_dim1 = x1.shape[1]
        H = weight_mat.shape[2]
        if shard_type == 0:
            C = x_dim1 // ep_ws
            reshape_1 = [E_div_ep, ep_ws * C, tp_ws, H // tp_ws]
            reshape_2 = [tp_ws * E_div_ep, ep_ws * C, H // tp_ws]
            reshape_3 = [E_div_ep, ep_ws, C, H // tp_ws]
            reshape_4 = [E_div_ep * ep_ws, C, H // tp_ws]
            tensor_scatter_shape = [E_div_ep, ep_ws * C, H // tp_ws]
            alltoall_shape = [ep_ws, E_div_ep, C, H // tp_ws]
        else:
            C_div_tp = x_dim1 // ep_ws // tp_ws
            reshape_1 = [E_div_ep, ep_ws, tp_ws, C_div_tp, H]
            reshape_2 = [tp_ws * E_div_ep, ep_ws * C_div_tp, H]
            reshape_3 = [E_div_ep, ep_ws, C_div_tp, H]
            reshape_4 = [E_div_ep * ep_ws, C_div_tp, H]
            tensor_scatter_shape = [E_div_ep, ep_ws * C_div_tp, H]
            alltoall_shape = [ep_ws, E_div_ep, C_div_tp, H]

        bmm_out = bmm_out.reshape(reshape_1)
        if shard_type == 0:
            bmm_out = bmm_out.permute(2, 0, 1, 3)
        else:
            bmm_out = bmm_out.permute(2, 0, 1, 3, 4)
        bmm_out = bmm_out.reshape(reshape_2).contiguous()

        rs_out = torch.zeros(tensor_scatter_shape, dtype=x1.dtype, device=f"npu:{rank}")
        dist._reduce_scatter_base(rs_out, bmm_out, op=ReduceOp.SUM, group=tp_group)
        torch.npu.synchronize()

        if is_bias and bias is not None:
            if bias.dim() == 2:
                bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
            rs_out = rs_out + bias

        rs_out = rs_out.reshape(reshape_3)
        rs_out = rs_out.permute(1, 0, 2, 3).contiguous()

        a2a_out = torch.zeros(alltoall_shape, dtype=x1.dtype, device=f"npu:{rank}")
        dist.all_to_all_single(a2a_out, rs_out, group=ep_group)
        torch.npu.synchronize()

        output = a2a_out.reshape(reshape_4)
        out_cpu = output.reshape(-1).contiguous().cpu()
        np.savez(
            f"{result_path}.did{rank}.npz", **{f"cascade_did{rank}": out_cpu.numpy()}
        )
        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
