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

"""Golden for aclnnBatchMatMulReduceScatterAlltoAll / mindspeed.npu_bmm_reducescatter_alltoall.

BatchMatMulReduceScatterAlltoAll: bmm -> reduce_scatter(TP) -> all_to_all(EP) -> output
Dual comm domain: TP groups do reduce_scatter, EP groups do all_to_all.
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
    from ttk.core_modules.npu.op_api.comparison import Comparator

    return mod.to_torch_f32, mod.fmt_compare_result, Comparator


_DTYPE_MAP = {
    "float16": None,
    "fp16": None,
    "float32": None,
    "fp32": None,
    "bfloat16": None,
    "bf16": None,
}


def _dtype_map():
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }


# ---------------------------------------------------------------------------
# input save / cascade
# ---------------------------------------------------------------------------


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
                else:
                    arrays[f"did{did}_t{i}"] = t.cpu().numpy()
                    dtypes[f"did{did}_t{i}_dtype"] = dtype_str
    arrays.update(dtypes)
    np.savez(path, **arrays)


_next_port_base = [30073]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30073
        return port


def run_bmm_rs_a2a_cascade(
    thread_contexts, device_ids, ep_ws, tp_ws, shard_type, is_trans=False, is_bias=False
):
    import subprocess
    import sys
    import torch

    n = len(device_ids)
    if n < 2:
        return {}
    if n != ep_ws * tp_ws:
        raise ValueError(f"BMM_RS_A2A: world_size={n} != ep_ws*tp_ws={ep_ws * tp_ws}")
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_bmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)
        np.savez(
            input_path + ".meta.npz",
            ep_ws=np.array([ep_ws], dtype=np.int64),
            tp_ws=np.array([tp_ws], dtype=np.int64),
            shard_type=np.array([shard_type], dtype=np.int64),
            is_trans=np.array([int(is_trans)], dtype=np.int64),
            is_bias=np.array([int(is_bias)], dtype=np.int64),
        )
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
                str(ep_ws),
                str(tp_ws),
                str(shard_type),
                str(int(is_trans)),
                str(int(is_bias)),
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
        first_ctx = thread_contexts[device_ids[0]]
        out_idx = first_ctx.output_tensor_indexes[0]
        out_shape = first_ctx.tensor_view_shapes[out_idx]
        result = {}
        for did in device_ids:
            main_t = None
            rank_file = f"{result_path}.did{did}.npz"
            if os.path.exists(rank_file):
                data = np.load(rank_file, allow_pickle=False)
                key = f"cascade_did{did}"
                if key in data.files:
                    main_t = torch.from_numpy(data[key].copy()).reshape(out_shape)
            result[did] = {"main": main_t}
        return result


# ---------------------------------------------------------------------------
# CPU golden
# ---------------------------------------------------------------------------


def bmm_rs_a2a_cpu_golden(thread_contexts, device_ids, ep_ws, tp_ws, world_size):
    import torch

    to_torch_f32, _, _ = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    shard_type = int(attrs.get("yShardType", 1))
    is_bias = bool(attrs.get("isBias", False))
    is_trans = bool(attrs.get("isTrans", False))

    in_dtype = thread_contexts[device_ids[0]].flatten_tensors[0].dtype
    E_div_ep = thread_contexts[device_ids[0]].flatten_tensors[0].shape[0]
    x_dim1 = thread_contexts[device_ids[0]].flatten_tensors[0].shape[1]
    H = thread_contexts[device_ids[0]].flatten_tensors[1].shape[2]
    if shard_type == 0:
        C = x_dim1 // ep_ws
    else:
        C_div_tp = x_dim1 // ep_ws // tp_ws

    rs_per_rank = {}
    n_ep_groups = world_size // ep_ws
    n_tp_groups = world_size // tp_ws
    # TP groups: contiguous (matching TTK profiling.py HCCL comm creation)
    # TP groups: [[0,1,2,3],[4,5,6,7]] for ep=2,tp=4,world=8
    for g in range(n_tp_groups):
        group_dids = [g * tp_ws + t for t in range(tp_ws)]
        all_parts = []
        for did in group_dids:
            tc = thread_contexts[did]
            x = to_torch_f32(tc.flatten_tensors[0])
            # fp32 bmm then truncate to bf16 (best CPU approximation of NPU bf16 Cube)
            weight = to_torch_f32(tc.flatten_tensors[1])
            bmm_out = torch.bmm(x, weight)
            bmm_out = bmm_out.to(in_dtype).float()
            del weight
            if shard_type == 0:
                r1 = bmm_out.reshape(E_div_ep, ep_ws * C, tp_ws, H // tp_ws)
                r1 = r1.permute(2, 0, 1, 3).contiguous()
                r1 = r1.reshape(tp_ws * E_div_ep, ep_ws * C, H // tp_ws)
            else:
                r1 = bmm_out.reshape(E_div_ep, ep_ws, tp_ws, C_div_tp, H)
                r1 = r1.permute(2, 0, 1, 3, 4).contiguous()
                r1 = r1.reshape(tp_ws * E_div_ep, ep_ws * C_div_tp, H)
            all_parts.append(r1.to(in_dtype))
            del bmm_out
        n_tp = len(group_dids)
        for local_idx, did in enumerate(group_dids):
            # reduce_scatter: cyclic accumulation with bf16 truncation after each step
            # (matching original TTK mc2_golden.py — simulates HCCL bf16 reduce)
            start = (local_idx + 1) % n_tp
            acc = (
                all_parts[start][local_idx * E_div_ep : (local_idx + 1) * E_div_ep]
                .clone()
                .float()
            )
            for step in range(1, n_tp):
                src_idx = (start + step) % n_tp
                src_chunk = all_parts[src_idx][
                    local_idx * E_div_ep : (local_idx + 1) * E_div_ep
                ]
                acc = acc.to(in_dtype).float() + src_chunk.float()
            chunk = acc.to(in_dtype)
            tc = thread_contexts[did]
            if is_bias and len(tc.flatten_tensors) > 2:
                bias = to_torch_f32(tc.flatten_tensors[2])
                if bias.numel() > 0:
                    if bias.dim() == 2:
                        bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
                    chunk = chunk.to(in_dtype).float() + bias
                    chunk = chunk.to(in_dtype)
                del bias
            rs_per_rank[did] = chunk
        del all_parts

    n_ep_groups = world_size // ep_ws
    rank_goldens = {}
    # EP groups: strided by tp_ws (matching TTK profiling.py HCCL comm creation)
    for g in range(n_ep_groups):
        group_dids = [g + e * tp_ws for e in range(ep_ws)]
        for target_local, target_did in enumerate(group_dids):
            if shard_type == 0:
                all_chunks = []
                for src_local, src_did in enumerate(group_dids):
                    rs = rs_per_rank[src_did]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C, H // tp_ws)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C, H // tp_ws)
            else:
                all_chunks = []
                for src_local, src_did in enumerate(group_dids):
                    rs = rs_per_rank[src_did]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C_div_tp, H)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C_div_tp, H)
            del gathered, all_chunks
            rank_goldens[target_did] = out
    del rs_per_rank
    return rank_goldens


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


def bmm_rs_a2a_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    _, fmt_compare, Comparator = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    world_size = len(device_ids)
    ep_ws = int(attrs.get("epWorldSize", 1))
    tp_ws = int(attrs.get("tpWorldSize", 1))
    shard_type = int(attrs.get("yShardType", 1))
    is_bias = bool(attrs.get("isBias", False))
    is_trans = bool(attrs.get("isTrans", False))
    dtype_map = _dtype_map()

    rank_third_parties = None
    try:
        cascade_outs = run_bmm_rs_a2a_cascade(
            thread_contexts,
            device_ids,
            ep_ws=ep_ws,
            tp_ws=tp_ws,
            shard_type=shard_type,
            is_trans=is_trans,
            is_bias=is_bias,
        )
        rank_third_parties = {did: [cascade_outs[did]["main"]] for did in device_ids}
        logging.info("BatchMatMulReduceScatterAlltoAll: real HCCL cascade succeeded")
    except Exception:
        logging.exception(
            "BatchMatMulReduceScatterAlltoAll: real HCCL cascade failed, no third_party"
        )
        rank_third_parties = None

    rank_goldens = bmm_rs_a2a_cpu_golden(
        thread_contexts, device_ids, ep_ws, tp_ws, world_size
    )

    for did in device_ids:
        tc = thread_contexts[did]
        out_idx = tc.output_tensor_indexes[0]
        out_shape = tc.tensor_view_shapes[out_idx]
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        golden = rank_goldens[did]
        if golden.shape != torch.Size(out_shape):
            golden = golden.reshape(out_shape)
        if len(out_dtypes) > 0:
            target_dtype = dtype_map.get(out_dtypes[0], None)
            if target_dtype is not None:
                golden = golden.to(target_dtype)
        tc.golden_tensors = [golden.contiguous()]
        del golden
        third_parties_list = None
        if rank_third_parties is not None:
            tp = rank_third_parties.get(did, [None])[0]
            if isinstance(tp, torch.Tensor):
                if tp.shape != torch.Size(out_shape):
                    tp = tp.reshape(out_shape)
                if len(out_dtypes) > 0:
                    target_dtype = dtype_map.get(out_dtypes[0], None)
                    if target_dtype is not None:
                        tp = tp.to(target_dtype)
                third_parties_list = [tp.contiguous()]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare(cr)})")
            if cr.passed != "PASS":
                logging.error(f"BMM_RS_A2A: rank dev={did} FAILED: {cr.precision}")
            else:
                logging.info(f"BMM_RS_A2A: rank dev={did} PASSED")
        except Exception:
            logging.exception(f"BMM_RS_A2A: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")
    del rank_goldens


def bmm_rs_a2a_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import numpy as _np

    shard_type = int(attrs.get("shard_type", attrs.get("yShardType", 0)))
    ep_ws = int(attrs.get("epWorldSize", 0))
    tp_ws = int(attrs.get("tpWorldSize", 0))
    if not ep_ws or not tp_ws:
        return None

    in_dtype = torch.float32
    if attrs and "_tensor_dtypes" in attrs:
        dt_str = (
            str(attrs["_tensor_dtypes"][0]) if attrs["_tensor_dtypes"] else "float32"
        )
        if "bfloat16" in dt_str or "bf16" in dt_str:
            in_dtype = torch.bfloat16
        elif "float16" in dt_str or "fp16" in dt_str:
            in_dtype = torch.float16
    E_div_ep = cpu_inputs_per_rank[0][0].shape[0]
    x_dim1 = cpu_inputs_per_rank[0][0].shape[1]
    H = cpu_inputs_per_rank[0][1].shape[2]
    is_bias = bool(attrs.get("isBias", False))
    if shard_type == 0:
        C = x_dim1 // ep_ws
    else:
        C_div_tp = x_dim1 // ep_ws // tp_ws

    rs_per_rank = {}
    n_ep_groups_e2e = ws // ep_ws
    n_tp_groups = ws // tp_ws
    for g in range(n_tp_groups):
        group_ranks = [g * tp_ws + t for t in range(tp_ws)]
        all_parts = []
        for r in group_ranks:
            x = cpu_inputs_per_rank[r][0].float()
            weight = cpu_inputs_per_rank[r][1].float()
            bmm_out = torch.bmm(x, weight)
            del weight
            if shard_type == 0:
                r1 = bmm_out.reshape(E_div_ep, ep_ws * C, tp_ws, H // tp_ws)
                r1 = r1.permute(2, 0, 1, 3).contiguous()
                r1 = r1.reshape(tp_ws * E_div_ep, ep_ws * C, H // tp_ws)
            else:
                r1 = bmm_out.reshape(E_div_ep, ep_ws, tp_ws, C_div_tp, H)
                r1 = r1.permute(2, 0, 1, 3, 4).contiguous()
                r1 = r1.reshape(tp_ws * E_div_ep, ep_ws * C_div_tp, H)
            all_parts.append(r1)
            del bmm_out
        n_tp = len(group_ranks)
        for local_idx, r in enumerate(group_ranks):
            acc = all_parts[0][
                local_idx * E_div_ep : (local_idx + 1) * E_div_ep
            ].clone()
            for src_idx in range(1, n_tp):
                src_chunk = all_parts[src_idx][
                    local_idx * E_div_ep : (local_idx + 1) * E_div_ep
                ]
                acc = acc + src_chunk
            chunk = acc
            if is_bias and len(cpu_inputs_per_rank[r]) > 2:
                bias = cpu_inputs_per_rank[r][2].float()
                if bias.numel() > 0:
                    if bias.dim() == 2:
                        bias = bias.reshape(bias.shape[0], 1, bias.shape[1])
                    chunk = chunk + bias
                del bias
            rs_per_rank[r] = chunk
        del all_parts

    n_ep_groups = ws // ep_ws
    for g in range(n_ep_groups):
        group_ranks = [g + e * tp_ws for e in range(ep_ws)]
        for target_local, target_rank in enumerate(group_ranks):
            if target_rank != rank:
                continue
            if shard_type == 0:
                all_chunks = []
                for src_local, src_rank in enumerate(group_ranks):
                    rs = rs_per_rank[src_rank]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C, H // tp_ws)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C, H // tp_ws)
            else:
                all_chunks = []
                for src_local, src_rank in enumerate(group_ranks):
                    rs = rs_per_rank[src_rank]
                    rs_r = rs.reshape(E_div_ep, ep_ws, C_div_tp, H)
                    rs_r = rs_r.permute(1, 0, 2, 3).contiguous()
                    all_chunks.append(rs_r[target_local].clone())
                gathered = torch.cat(all_chunks, dim=0)
                out = gathered.reshape(E_div_ep * ep_ws, C_div_tp, H)
            del gathered, all_chunks
            return [out]
    return None
