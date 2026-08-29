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

"""Standalone cascade worker for MatmulReduceScatter.

Args: rank world_size port input_path result_path error_path
      is_trans_b m_dim k_dim n_dim
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
        is_trans_b,
        m_dim,
        k_dim,
        n_dim,
    ) = sys.argv[1:11]
    rank = int(rank)
    world_size = int(world_size)
    m_dim = int(m_dim)
    n_dim = int(n_dim)
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
        w_dtype_str = str(x2.dtype).replace("torch.", "")
        if any(d in w_dtype_str for d in ("float8", "hifloat8", "hif8", "fp8", "int8")):
            mm_out = torch.matmul(x1.float(), x2.float())
        else:
            mm_out = torch.matmul(x1, x2)
        m_chunk = m_dim // world_size
        scatter = torch.empty(m_chunk, n_dim, dtype=mm_out.dtype, device=f"npu:{rank}")
        dist._reduce_scatter_base(scatter, mm_out, op=ReduceOp.SUM)
        torch.npu.synchronize()
        sc_cpu = scatter.reshape(-1).contiguous().cpu()
        rank_file = f"{result_path}.did{rank}.npz"
        np.savez(rank_file, **{f"cascade_did{rank}": sc_cpu.numpy()})
        dist.destroy_process_group()
    except Exception:
        tb = traceback.format_exc()
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{tb}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
