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

"""Standalone cascade worker for AlltoAllMatmul.

Launched as a subprocess so it does not depend on pickle of dynamically-loaded
spec modules. Mirrors TTK's E2E worker pattern.

Args: rank world_size port input_path result_path error_path
      transpose_x1 transpose_x2 mm_m_chunk k_dim n_dim is_alltoall_output
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
            elif "float8_e4m3" in dtype_str:
                t = (
                    t.view(torch.uint8).to(torch.float8_e4m3fn)
                    if hasattr(torch, "float8_e4m3fn")
                    else t
                )
            elif "float8_e5m2" in dtype_str:
                t = (
                    t.view(torch.uint8).to(torch.float8_e5m2)
                    if hasattr(torch, "float8_e5m2")
                    else t
                )
            elif "hifloat8" in dtype_str:
                t = t.view(torch.uint8)
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
        transpose_x1,
        transpose_x2,
        mm_m_chunk,
        k_dim,
        n_dim,
        is_alltoall_output,
    ) = sys.argv[1:13]
    rank = int(rank)
    world_size = int(world_size)
    mm_m_chunk = int(mm_m_chunk)
    k_dim = int(k_dim)
    n_dim = int(n_dim)
    transpose_x1 = transpose_x1 == "True"
    transpose_x2 = transpose_x2 == "True"
    is_alltoall_output = is_alltoall_output == "True"
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
        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        bias = (
            tensors[2].npu()
            if 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0
            else None
        )
        input_mat = x1.t().contiguous() if transpose_x1 else x1
        weight_mat = x2.t().contiguous() if transpose_x2 else x2
        input_re = input_mat.reshape(world_size, mm_m_chunk, k_dim).contiguous()
        alltoall_out = torch.empty(
            world_size, mm_m_chunk, k_dim, dtype=input_re.dtype, device=f"npu:{rank}"
        )
        dist.all_to_all_single(alltoall_out, input_re)
        torch.npu.synchronize()
        a2a_out = (
            alltoall_out.permute(1, 0, 2)
            .reshape(mm_m_chunk, world_size * k_dim)
            .contiguous()
        )
        w_dtype_str = str(weight_mat.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            mm_out = torch.matmul(a2a_out.float(), weight_mat.float())
        else:
            mm_out = torch.matmul(a2a_out, weight_mat)
        if bias is not None:
            mm_out = mm_out + bias
        mm_cpu = mm_out.reshape(-1).contiguous().cpu()
        rank_file = f"{result_path}.did{rank}.npz"
        np.savez(rank_file, **{f"cascade_did{rank}": mm_cpu.numpy()})
        if is_alltoall_output:
            a2a_cpu = a2a_out.reshape(-1).contiguous().cpu()
            a2a_file = f"{result_path}.a2a_did{rank}.npz"
            np.savez(a2a_file, **{f"cascade_a2a_did{rank}": a2a_cpu.numpy()})
        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
