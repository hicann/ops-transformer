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

"""Standalone cascade worker for AlltoAllAllGatherBatchMatMul.

Flow: all_to_all(EP) -> all_gather(TP) -> bmm -> [bias + act] -> output
Dual comm domain: EP groups do all_to_all, TP groups do all_gather.
Args: rank world_size port input_path result_path error_path
      ep_ws tp_ws shard_type is_trans is_bias act_type need_ag_out need_act_feat
"""

import os
import sys
import datetime
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist


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


def _activate_npu(x, act_type):
    if act_type == 0:
        return x
    elif act_type == 1:
        return torch.nn.functional.gelu(x)
    elif act_type == 2:
        return torch.nn.functional.silu(x)
    elif act_type == 3:
        return torch.relu(x)
    elif act_type == 4:
        return x / (1.0 + torch.exp(-1.702 * x))
    return x


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
        act_type,
        need_ag_out,
        need_act_feat,
    ) = sys.argv[1:16]
    rank = int(rank)
    world_size = int(world_size)
    ep_ws = int(ep_ws)
    tp_ws = int(tp_ws)
    shard_type = int(shard_type)
    is_trans = is_trans == "True"
    is_bias = is_bias == "True"
    act_type = int(act_type)
    need_ag_out = need_ag_out == "True"
    need_act_feat = need_act_feat == "True"
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
        weight = tensors[1].npu()
        bias = None
        if (
            is_bias
            and 2 in tensors
            and tensors[2] is not None
            and tensors[2].numel() > 0
        ):
            bias = tensors[2].npu()

        E = x1.shape[0]
        E_div_ep = E // ep_ws
        if shard_type == 0:
            C = x1.shape[1]
            H_div_tp = x1.shape[2]
            H = H_div_tp * tp_ws
            reshape_1 = [ep_ws, E_div_ep, C, H_div_tp]
            tensor_ag_shape = [tp_ws * E_div_ep, ep_ws, C, H_div_tp]
            reshape_2 = [tp_ws, E_div_ep, ep_ws, C, H_div_tp]
            reshape_3 = [E_div_ep, ep_ws * C, H]
        else:
            C_div_tp = x1.shape[1]
            H = x1.shape[2]
            C = C_div_tp * tp_ws
            reshape_1 = [ep_ws, E_div_ep, C_div_tp, H]
            tensor_ag_shape = [tp_ws * E_div_ep, ep_ws, C_div_tp, H]
            reshape_2 = [tp_ws, E_div_ep, ep_ws, C_div_tp, H]
            reshape_3 = [E_div_ep, ep_ws * C, H]

        a2a_out = torch.zeros_like(x1)
        dist.all_to_all_single(a2a_out, x1, group=ep_group)
        torch.npu.synchronize()
        a2a_out = a2a_out.reshape(reshape_1).permute(1, 0, 2, 3).contiguous()

        ag_out = torch.zeros(tensor_ag_shape, dtype=x1.dtype, device=f"npu:{rank}")
        dist._all_gather_base(ag_out, a2a_out, group=tp_group)
        torch.npu.synchronize()

        ag_out = ag_out.reshape(reshape_2)
        if shard_type == 0:
            ag_out = ag_out.permute(1, 2, 3, 0, 4).contiguous()
        else:
            ag_out = ag_out.permute(1, 2, 0, 3, 4).contiguous()
        gather_output = ag_out.reshape(reshape_3)

        bmm_out = torch.bmm(gather_output, weight)
        if is_bias and bias is not None:
            if bias.dim() == 2:
                bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
            bmm_out = bmm_out + bias
        act_out = _activate_npu(bmm_out, act_type)

        main_cpu = act_out.reshape(-1).contiguous().cpu()
        np.savez(
            f"{result_path}.did{rank}.npz", **{f"cascade_did{rank}": main_cpu.numpy()}
        )
        if need_ag_out:
            ag_cpu = gather_output.reshape(-1).contiguous().cpu()
            np.savez(
                f"{result_path}.a2a_did{rank}.npz",
                **{f"cascade_a2a_did{rank}": ag_cpu.numpy()},
            )
        if need_act_feat:
            bmm_cpu = bmm_out.reshape(-1).contiguous().cpu()
            np.savez(
                f"{result_path}.mm_did{rank}.npz",
                **{f"cascade_mm_did{rank}": bmm_cpu.numpy()},
            )
        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
