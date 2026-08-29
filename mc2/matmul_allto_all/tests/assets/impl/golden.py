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

"""Golden for aclnnMatmulAlltoAll / torch_npu.npu_matmul_all_to_all.

aclnnMatmulAlltoAll: matmul(x1, x2) -> all_to_all -> output
torch_npu.npu_matmul_all_to_all: same flow, E2E path.

CPU golden (cross_check third_party): pure fp32 matmul + CPU-simulated
all_to_all, matching mc2_test op_class get_cpu.
HCCL cascade third_party: fork multi-process + real HCCL all_to_all_single,
matching mc2_test op_class get_hccl_mm.
"""

import importlib.util
import logging
import os
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# lazy framework helpers (TTK side)
# ---------------------------------------------------------------------------


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
    return mod.to_torch_f32, mod.fmt_compare_result, mod.apply_a2a_goldens_and_compare


# ---------------------------------------------------------------------------
# CPU golden (pure fp32 + simulated all_to_all)
# ---------------------------------------------------------------------------


def matmul_allto_all_cpu_golden(thread_contexts, device_ids, world_size):
    """Compute per-rank golden dict for MatmulAlltoAll.

    matmul(x1, x2) -> chunk along N -> all_to_all -> output [M*ws, chunk_n]
    """
    import torch

    to_torch_f32, _, _ = _load_framework()

    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    t_x1 = bool(attrs.get("transposeX1", False))
    t_x2 = bool(attrs.get("transposeX2", False))
    # determine M, N, chunk_n from first rank (all ranks share weight)
    tc0 = thread_contexts[device_ids[0]]
    x1_0 = tc0.flatten_tensors[0]
    x2_0 = tc0.flatten_tensors[1]
    mm_m = x1_0.shape[1] if t_x1 else x1_0.shape[0]
    mm_n = x2_0.shape[0] if t_x2 else x2_0.shape[1]
    chunk_n = mm_n // world_size

    rank_goldens = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        all_to_all_results = []
        for src_did in device_ids:
            src_tc = thread_contexts[src_did]
            src_x1 = src_tc.flatten_tensors[0]
            src_x2 = src_tc.flatten_tensors[1]
            src_bias = (
                src_tc.flatten_tensors[2] if len(src_tc.flatten_tensors) > 2 else None
            )
            if (
                src_bias is not None
                and isinstance(src_bias, torch.Tensor)
                and src_bias.numel() == 0
            ):
                src_bias = None
            s_input = to_torch_f32(src_x1)
            if t_x1:
                s_input = s_input.t().contiguous()
            s_weight = to_torch_f32(src_x2)
            if t_x2:
                s_weight = s_weight.t().contiguous()
            s_mm = torch.matmul(s_input, s_weight)
            if src_bias is not None:
                s_mm = s_mm + to_torch_f32(src_bias)
            s_chunks = (
                s_mm.view(mm_m, world_size, chunk_n).permute(1, 0, 2).contiguous()
            )
            s_chunks = s_chunks.view(world_size, mm_m * chunk_n)
            send_chunks = s_chunks.chunk(world_size, dim=0)
            all_to_all_results.append(send_chunks[target_idx].clone())
            del s_mm, s_chunks, send_chunks
        received = torch.cat(all_to_all_results, dim=0)
        received = received.reshape(-1, chunk_n).contiguous()
        rank_goldens[target_did] = {"main": received}
        del all_to_all_results, received
    return rank_goldens


# ---------------------------------------------------------------------------
# HCCL cascade (real NPU matmul + real HCCL all_to_all_single)
# ---------------------------------------------------------------------------


def _save_inputs_per_rank(thread_contexts, device_ids, path):
    import torch

    arrays = {}
    dtypes = {}
    for did in device_ids:
        tc = thread_contexts[did]
        flat = tc.flatten_tensors
        for i, t in enumerate(flat):
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


_next_port_base = [30000]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30000
        return port


def run_matmul_alltoall_cascade(
    thread_contexts, device_ids, transpose_x1=False, transpose_x2=False
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
    mm_m = x1.shape[1] if transpose_x1 else x1.shape[0]
    mm_n = x2.shape[0] if transpose_x2 else x2.shape[1]
    chunk_n = mm_n // n
    if chunk_n * n != mm_n:
        raise ValueError(f"MatmulAlltoAll: mm_n={mm_n} not divisible by world_size={n}")
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_") as tmpdir:
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
                str(transpose_x1),
                str(transpose_x2),
                str(mm_m),
                str(chunk_n),
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


# ---------------------------------------------------------------------------
# ACLNN multi-device golden entry (called by TTK framework)
# ---------------------------------------------------------------------------


def matmul_allto_all_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    world_size = len(device_ids)
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    t_x1 = bool(attrs.get("transposeX1", False))
    t_x2 = bool(attrs.get("transposeX2", False))
    rank_goldens = matmul_allto_all_cpu_golden(thread_contexts, device_ids, world_size)

    rank_third_parties = None
    try:
        cascade_outs = run_matmul_alltoall_cascade(
            thread_contexts, device_ids, transpose_x1=t_x1, transpose_x2=t_x2
        )
        rank_third_parties = {did: [cascade_outs[did]] for did in device_ids}
        logging.info("MatmulAlltoAll: real HCCL cascade succeeded")
    except Exception:
        logging.exception("MatmulAlltoAll: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    _, _, apply_a2a = _load_framework()
    apply_a2a(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


# ---------------------------------------------------------------------------
# E2E multi-device golden entry (called by E2E worker)
# ---------------------------------------------------------------------------


def matmul_all_to_all_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    world_size = len(cpu_inputs_per_rank)
    t_x1 = bool(attrs.get("transposeX1", False))
    t_x2 = bool(attrs.get("transposeX2", False))
    all_mm_out = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0]
        x2 = cpu_inputs_per_rank[r][1]
        bias = cpu_inputs_per_rank[r][2] if len(cpu_inputs_per_rank[r]) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        input_mat = x1.t().contiguous() if t_x1 else x1
        weight_mat = x2.t().contiguous() if t_x2 else x2
        mm_out = torch.matmul(input_mat.float(), weight_mat.float())
        if bias is not None:
            mm_out = mm_out + bias.float()
        all_mm_out.append(mm_out)

    M = all_mm_out[0].shape[0]
    N = all_mm_out[0].shape[1]
    chunk_n = N // world_size
    goldens = []
    for target_r in range(world_size):
        parts = []
        for src_r in range(world_size):
            s_mm = all_mm_out[src_r]
            s_chunks = s_mm.view(M, world_size, chunk_n).permute(1, 0, 2).contiguous()
            s_chunks = s_chunks.view(world_size, M * chunk_n)
            send_chunks = s_chunks.chunk(world_size, dim=0)
            parts.append(send_chunks[target_r].clone())
            del s_chunks, send_chunks
        received = torch.cat(parts, dim=0).reshape(-1, chunk_n).contiguous()
        goldens.append(received)
        del parts
    del all_mm_out
    return goldens
