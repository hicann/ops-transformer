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

"""Golden for aclnnMatmulReduceScatter / torch_npu.npu_mm_reduce_scatter_base.

matmul(x1, x2) -> reduce_scatter(SUM) -> output [M/ws, N]
"""

import logging
import os
import tempfile
from pathlib import Path

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
    return mod.to_torch_f32, mod.apply_goldens_and_compare


def _save_inputs_per_rank(thread_contexts, device_ids, path):
    import torch

    arrays = {}
    dtypes = {}
    for did in device_ids:
        tc = thread_contexts[did]
        for i, t in enumerate(tc.flatten_tensors):
            if t is None:
                continue
            if isinstance(t, torch.Tensor):
                dtype_str = str(t.dtype).replace("torch.", "")
                if t.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    arrays[f"did{did}_t{i}"] = t.float().cpu().numpy()
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
                elif "float8" in dtype_str or "hifloat8" in dtype_str:
                    arrays[f"did{did}_t{i}"] = t.view(torch.uint8).cpu().numpy()
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
                else:
                    arrays[f"did{did}_t{i}"] = t.cpu().numpy()
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
            else:
                arrays[f"did{did}_t{i}"] = np.asarray(t)
                dtypes[f"did{did}_t{i}_dtype"] = "numpy"
    arrays.update(dtypes)
    np.savez(path, **arrays)


_next_port_base = [30026]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30026
        return port


def matmul_reduce_scatter_cpu_golden(thread_contexts, device_ids, world_size):
    import torch

    to_torch_f32, _ = _load_framework()
    # Match NPU bf16/fp16 accumulation: fp32 matmul -> dtype truncation -> dtype accumulate
    first_ctx = next(iter(thread_contexts.values()))
    orig_dtype = torch.float32
    x1_dtype_str = (
        first_ctx.flat_tensor_dtypes[0] if first_ctx.flat_tensor_dtypes else ""
    )
    if "bfloat16" in x1_dtype_str or "bf16" in x1_dtype_str:
        orig_dtype = torch.bfloat16
    elif "float16" in x1_dtype_str or "fp16" in x1_dtype_str:
        orig_dtype = torch.float16

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        mm_out = torch.matmul(to_torch_f32(x1), to_torch_f32(x2))
        mm_out = mm_out.to(orig_dtype).float()
        if bias is not None:
            mm_out = mm_out + to_torch_f32(bias)
        local_results[did] = mm_out.to(orig_dtype)
    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results
    M = total.shape[0]
    chunk_m = M // world_size
    rank_goldens = {}
    for idx, did in enumerate(device_ids):
        rank_goldens[did] = total[idx * chunk_m : (idx + 1) * chunk_m, :].contiguous()
    del total
    return rank_goldens


def run_matmul_reducescatter_cascade(thread_contexts, device_ids, is_trans_b=False):
    import subprocess
    import sys
    import torch

    n = len(device_ids)
    if n < 2:
        return {}
    first_ctx = thread_contexts[device_ids[0]]
    x1 = first_ctx.flatten_tensors[0]
    x2 = first_ctx.flatten_tensors[1]
    m_dim = x1.shape[0]
    n_dim = x2.shape[1]
    if m_dim % n != 0:
        raise ValueError(f"MatmulReduceScatter: M={m_dim} not divisible by ws={n}")
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_rsmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)
        procs = []
        for rank in range(n):
            cmd = [
                sys.executable,
                worker_script,
                str(rank),
                str(n),
                str(port),
                input_path,
                result_path,
                error_path,
                str(is_trans_b),
                str(m_dim),
                str(x1.shape[1]),
                str(n_dim),
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            procs.append(p)
        for p in procs:
            p.wait()
        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for p in procs:
            if p.returncode != 0:
                stderr = (
                    p.stderr.read().decode(errors="replace")[-2000:] if p.stderr else ""
                )
                raise RuntimeError(
                    f"cascade worker exited {p.returncode}\n{error_msg}\n{stderr}"
                )
        outs = {}
        for did in device_ids:
            rank_file = f"{result_path}.did{did}.npz"
            if os.path.exists(rank_file):
                data = np.load(rank_file, allow_pickle=False)
                key = f"cascade_did{did}"
                if key in data.files:
                    outs[did] = torch.from_numpy(data[key].copy())
        if len(outs) != n:
            raise RuntimeError(
                f"cascade result incomplete: {len(outs)}/{n}\n{error_msg}"
            )
        return outs


def matmul_reduce_scatter_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch

    world_size = len(device_ids)
    first_ctx = next(iter(thread_contexts.values()))
    remark = first_ctx.remark or ""
    is_trans_b = False
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "is_trans_b":
            is_trans_b = kv[1].strip() == "1"

    rank_goldens = matmul_reduce_scatter_cpu_golden(
        thread_contexts, device_ids, world_size
    )

    rank_third_parties = None
    try:
        cascade_outs = run_matmul_reducescatter_cascade(
            thread_contexts, device_ids, is_trans_b=is_trans_b
        )
        rank_third_parties = {did: [cascade_outs[did]] for did in device_ids}
        logging.info("MatmulReduceScatter: real HCCL cascade succeeded")
    except Exception:
        logging.exception(
            "MatmulReduceScatter: real HCCL cascade failed, no third_party"
        )
        rank_third_parties = None

    _, apply_goldens = _load_framework()
    apply_goldens(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


def mm_reduce_scatter_base_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    orig_dtype = torch.float32
    if attrs and "_tensor_dtypes" in attrs:
        dt_str = (
            str(attrs["_tensor_dtypes"][0]) if attrs["_tensor_dtypes"] else "float32"
        )
        if dt_str == "bfloat16":
            orig_dtype = torch.bfloat16
        elif dt_str == "float16":
            orig_dtype = torch.float16
    local_results = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0]
        x2 = cpu_inputs_per_rank[r][1]
        mm_out = torch.matmul(x1.float(), x2.float())
        mm_out = mm_out.to(orig_dtype).float()
        local_results.append(mm_out.to(orig_dtype))
    total = torch.zeros_like(local_results[0])
    for lr in local_results:
        total = total + lr
    total = total.float()
    del local_results
    M = total.shape[0]
    chunk_m = M // world_size
    return [
        total[r * chunk_m : (r + 1) * chunk_m, :].contiguous()
        for r in range(world_size)
    ]
