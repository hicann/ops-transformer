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

"""Golden for aclnnAlltoAllMatmul / torch_npu.npu_all_to_all_matmul.

aclnnAlltoAllMatmul: all_to_all(x1) -> matmul(a2a_out, x2) -> output [+ alltoall_output]
torch_npu.npu_all_to_all_matmul: same flow, E2E path.
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
    return mod.to_torch_f32, mod.apply_a2a_goldens_and_compare


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


_next_port_base = [30013]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30013
        return port


def allto_all_matmul_cpu_golden(thread_contexts, device_ids, world_size):
    import torch

    to_torch_f32, _ = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    t_x1 = bool(attrs.get("transposeX1", False))
    t_x2 = bool(attrs.get("transposeX2", False))

    rank_goldens = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        tc = thread_contexts[target_did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        weight_mat = to_torch_f32(x2)
        if t_x2:
            weight_mat = weight_mat.t().contiguous()
        M_total = (
            to_torch_f32(x1).shape[0] if not t_x1 else to_torch_f32(x1).t().shape[0]
        )
        # recompute properly
        input_mat = to_torch_f32(x1)
        if t_x1:
            input_mat = input_mat.t().contiguous()
        M_total = input_mat.shape[0]
        K = input_mat.shape[1]
        M_chunk = M_total // world_size

        recv_chunks = []
        for src_did in device_ids:
            src_tc = thread_contexts[src_did]
            src_x1 = src_tc.flatten_tensors[0]
            s_input = to_torch_f32(src_x1)
            if t_x1:
                s_input = s_input.t().contiguous()
            s_reshaped = s_input.view(world_size, M_chunk, K)
            recv_chunks.append(s_reshaped[target_idx])
        recv_tensor = torch.stack(recv_chunks, dim=0)
        a2a_out = (
            recv_tensor.permute(1, 0, 2).reshape(M_chunk, world_size * K).contiguous()
        )
        mm_out = torch.matmul(a2a_out, weight_mat)
        if bias is not None:
            mm_out = mm_out + to_torch_f32(bias)
        rank_goldens[target_did] = {"main": mm_out, "alltoall": a2a_out}
        del recv_chunks, recv_tensor, a2a_out, mm_out
    return rank_goldens


def run_alltoall_matmul_cascade(
    thread_contexts,
    device_ids,
    transpose_x1=False,
    transpose_x2=False,
    is_alltoall_output=False,
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
    input_mat_shape = (
        (x1.shape[1], x1.shape[0]) if transpose_x1 else (x1.shape[0], x1.shape[1])
    )
    weight_mat_shape = (
        (x2.shape[1], x2.shape[0]) if transpose_x2 else (x2.shape[0], x2.shape[1])
    )
    mm_m_chunk = input_mat_shape[0] // n
    k_dim = input_mat_shape[1]
    n_dim = weight_mat_shape[1]
    if mm_m_chunk * n != input_mat_shape[0]:
        raise ValueError(
            f"AlltoAllMatmul: M_total={input_mat_shape[0]} not divisible by ws={n}"
        )
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_a2amm_") as tmpdir:
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
                str(mm_m_chunk),
                str(k_dim),
                str(n_dim),
                str(is_alltoall_output),
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
            rank_file = f"{result_path}.did{did}.npz"
            main_t = None
            if os.path.exists(rank_file):
                data = np.load(rank_file, allow_pickle=False)
                key = f"cascade_did{did}"
                if key in data.files:
                    main_t = torch.from_numpy(data[key].copy()).reshape(
                        mm_m_chunk, n_dim
                    )
            a2a_t = None
            a2a_file = f"{result_path}.a2a_did{did}.npz"
            if os.path.exists(a2a_file):
                data = np.load(a2a_file, allow_pickle=False)
                key = f"cascade_a2a_did{did}"
                if key in data.files:
                    a2a_t = torch.from_numpy(data[key].copy()).reshape(
                        mm_m_chunk, n * k_dim
                    )
            result[did] = {"main": main_t, "alltoall": a2a_t}
        return result


def allto_all_matmul_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    world_size = len(device_ids)
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    t_x1 = bool(attrs.get("transposeX1", False))
    t_x2 = bool(attrs.get("transposeX2", False))
    first_tc = next(iter(thread_contexts.values()))
    is_a2a_out = len(first_tc.output_tensor_indexes) >= 2

    rank_goldens = allto_all_matmul_cpu_golden(thread_contexts, device_ids, world_size)

    rank_third_parties = None
    try:
        cascade_outs = run_alltoall_matmul_cascade(
            thread_contexts,
            device_ids,
            transpose_x1=t_x1,
            transpose_x2=t_x2,
            is_alltoall_output=is_a2a_out,
        )
        rank_third_parties = {
            did: [cascade_outs[did]["main"], cascade_outs[did].get("alltoall")]
            for did in device_ids
        }
        logging.info("AlltoAllMatmul: real HCCL cascade succeeded")
    except Exception:
        logging.exception("AlltoAllMatmul: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    _, apply_a2a = _load_framework()
    apply_a2a(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


def all_to_all_matmul_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    world_size = len(cpu_inputs_per_rank)
    t_x1 = bool(attrs.get("transposeX1", False))
    t_x2 = bool(attrs.get("transposeX2", False))
    input_mats = []
    weight_mats = []
    biases = []
    for r in range(world_size):
        x1 = cpu_inputs_per_rank[r][0]
        x2 = cpu_inputs_per_rank[r][1]
        bias = cpu_inputs_per_rank[r][2] if len(cpu_inputs_per_rank[r]) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        input_mats.append((x1.t().contiguous() if t_x1 else x1).float())
        weight_mats.append((x2.t().contiguous() if t_x2 else x2).float())
        biases.append(bias.float() if bias is not None else None)
    M_total = input_mats[0].shape[0]
    K = input_mats[0].shape[1]
    M_chunk = M_total // world_size
    goldens = []
    for target_r in range(world_size):
        recv_chunks = []
        for src_r in range(world_size):
            s_input = input_mats[src_r]
            s_reshaped = s_input.view(world_size, M_chunk, K)
            recv_chunks.append(s_reshaped[target_r])
        recv_tensor = torch.stack(recv_chunks, dim=0)
        a2a_out = (
            recv_tensor.permute(1, 0, 2).reshape(M_chunk, world_size * K).contiguous()
        )
        weight_mat = weight_mats[target_r]
        mm_out = torch.matmul(a2a_out, weight_mat)
        bias = biases[target_r]
        if bias is not None:
            mm_out = mm_out + bias
        goldens.append(mm_out)
    del input_mats, weight_mats, biases
    return goldens
