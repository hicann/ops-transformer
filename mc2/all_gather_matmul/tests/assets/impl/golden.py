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

"""Golden for aclnnAllGatherMatmul / torch_npu.npu_all_gather_base_mm.

all_gather(x1) -> matmul(gathered, x2) -> output [+ gather_output]
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


_next_port_base = [30039]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30039
        return port


def _get_orig_dtype(first_ctx):
    import torch

    x1_dtype_str = (
        first_ctx.flat_tensor_dtypes[0] if first_ctx.flat_tensor_dtypes else ""
    )
    if "bfloat16" in x1_dtype_str or "bf16" in x1_dtype_str:
        return torch.bfloat16
    if "float16" in x1_dtype_str or "fp16" in x1_dtype_str:
        return torch.float16
    return torch.float32


def all_gather_matmul_cpu_golden(thread_contexts, device_ids, world_size):
    import torch

    to_torch_f32, _ = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    orig_dtype = _get_orig_dtype(first_ctx)
    gather_output = len(first_ctx.output_tensor_indexes or ()) > 1

    all_x1 = [
        to_torch_f32(thread_contexts[did].flatten_tensors[0]) for did in device_ids
    ]
    gathered = torch.cat(all_x1, dim=0)
    del all_x1

    rank_goldens = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x2 = to_torch_f32(tc.flatten_tensors[1])
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        golden = torch.matmul(gathered, x2)
        if orig_dtype != torch.float32:
            golden = golden.to(orig_dtype).float()
        if bias is not None:
            golden = golden + to_torch_f32(bias)
        if gather_output:
            rank_goldens[did] = {
                "main": golden.contiguous(),
                "gather": gathered.contiguous(),
            }
        else:
            rank_goldens[did] = golden.contiguous()
        del golden
    del gathered
    return rank_goldens


def run_allgather_matmul_cascade(
    thread_contexts, device_ids, is_trans_b=False, is_gather_output=False
):
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
    k_dim = x1.shape[1]
    n_dim = x2.shape[1]
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_agmm_") as tmpdir:
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
                str(k_dim),
                str(n_dim),
                str(is_gather_output),
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
        result = {}
        for did in device_ids:
            main_t = None
            rank_file = f"{result_path}.did{did}.npz"
            if os.path.exists(rank_file):
                data = np.load(rank_file, allow_pickle=False)
                key = f"cascade_did{did}"
                if key in data.files:
                    main_t = torch.from_numpy(data[key].copy())
            gather_t = None
            a2a_file = f"{result_path}.a2a_did{did}.npz"
            if os.path.exists(a2a_file):
                data = np.load(a2a_file, allow_pickle=False)
                key = f"cascade_a2a_did{did}"
                if key in data.files:
                    gather_t = torch.from_numpy(data[key].copy())
            result[did] = {"main": main_t, "gather": gather_t}
        return result


def all_gather_matmul_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    world_size = len(device_ids)
    first_ctx = next(iter(thread_contexts.values()))
    remark = first_ctx.remark or ""
    is_trans_b = False
    gather_output = len(first_ctx.output_tensor_indexes or ()) > 1
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2:
            if kv[0].strip() == "is_trans_b":
                is_trans_b = kv[1].strip() == "1"
            elif kv[0].strip() == "gather_output":
                gather_output = kv[1].strip() == "1"

    rank_goldens = all_gather_matmul_cpu_golden(thread_contexts, device_ids, world_size)

    rank_third_parties = None
    try:
        cascade_outs = run_allgather_matmul_cascade(
            thread_contexts,
            device_ids,
            is_trans_b=is_trans_b,
            is_gather_output=gather_output,
        )
        rank_third_parties = {}
        for did in device_ids:
            tp_list = [cascade_outs[did]["main"]]
            if gather_output and cascade_outs[did].get("gather") is not None:
                tp_list.append(cascade_outs[did]["gather"])
            rank_third_parties[did] = tp_list
        logging.info("AllGatherMatmul: real HCCL cascade succeeded")
    except Exception:
        logging.exception("AllGatherMatmul: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    _, apply_goldens = _load_framework()
    apply_goldens(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


def all_gather_base_mm_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    world_size = len(cpu_inputs_per_rank)
    all_x1 = [inp[0].float() for inp in cpu_inputs_per_rank]
    gathered = torch.cat(all_x1, dim=0)
    del all_x1
    goldens = []
    for r in range(world_size):
        x2 = cpu_inputs_per_rank[r][1].float()
        g = torch.matmul(gathered, x2)
        goldens.append(g.contiguous())
    del gathered
    return goldens
