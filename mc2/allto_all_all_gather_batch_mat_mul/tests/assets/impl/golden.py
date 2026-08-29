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

"""Golden for aclnnAlltoAllAllGatherBatchMatMul / mindspeed.npu_alltoall_allgather_bmm.

AlltoAllAllGatherBatchMatMul: all_to_all(EP) -> all_gather(TP) -> bmm -> [bias + act] -> output
Dual comm domain: EP groups do all_to_all, TP groups do all_gather.
Multi-output: main (activated), allgather (gathered input), bmm (before activation).
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


def _apply_act(x, act):
    import torch

    if act == 0:
        return x
    elif act == 1:
        return torch.nn.functional.gelu(x)
    elif act == 2:
        return torch.nn.functional.silu(x)
    elif act == 3:
        return torch.nn.functional.relu(x)
    elif act == 4:
        return x / (1.0 + torch.exp(-1.702 * x))
    return x


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


_next_port_base = [30086]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30086
        return port


def run_a2a_ag_bmm_cascade(
    thread_contexts,
    device_ids,
    ep_ws,
    tp_ws,
    shard_type,
    is_trans=False,
    is_bias=False,
    act_type=0,
    need_ag_out=True,
    need_act_feat=False,
):
    import subprocess
    import sys
    import torch

    n = len(device_ids)
    if n < 2:
        return {}
    if n != ep_ws * tp_ws:
        raise ValueError(f"A2A_AG_BMM: world_size={n} != ep_ws*tp_ws={ep_ws * tp_ws}")
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_a2aagbmm_") as tmpdir:
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
            act_type=np.array([act_type], dtype=np.int64),
            need_ag_out=np.array([int(need_ag_out)], dtype=np.int64),
            need_act_feat=np.array([int(need_act_feat)], dtype=np.int64),
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
                str(is_trans),
                str(is_bias),
                str(act_type),
                str(need_ag_out),
                str(need_act_feat),
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
        out_idxs = first_ctx.output_tensor_indexes
        out_shapes = [first_ctx.tensor_view_shapes[oi] for oi in out_idxs]
        result = {}
        for did in device_ids:
            main_t = None
            rank_file = f"{result_path}.did{did}.npz"
            if os.path.exists(rank_file):
                data = np.load(rank_file, allow_pickle=False)
                key = f"cascade_did{did}"
                if key in data.files:
                    main_t = torch.from_numpy(data[key].copy()).reshape(out_shapes[0])
            ag_t = None
            if need_ag_out and len(out_shapes) > 1:
                ag_file = f"{result_path}.a2a_did{did}.npz"
                if os.path.exists(ag_file):
                    data = np.load(ag_file, allow_pickle=False)
                    key = f"cascade_a2a_did{did}"
                    if key in data.files:
                        ag_t = torch.from_numpy(data[key].copy()).reshape(out_shapes[1])
            bmm_t = None
            if need_act_feat and len(out_shapes) > 2:
                bmm_file = f"{result_path}.mm_did{did}.npz"
                if os.path.exists(bmm_file):
                    data = np.load(bmm_file, allow_pickle=False)
                    key = f"cascade_mm_did{did}"
                    if key in data.files:
                        bmm_t = torch.from_numpy(data[key].copy()).reshape(
                            out_shapes[2]
                        )
            result[did] = {"main": main_t, "allgather": ag_t, "bmm": bmm_t}
        return result


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


def a2a_ag_bmm_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    to_torch_f32, fmt_compare, Comparator = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    world_size = len(device_ids)
    ep_ws = int(attrs.get("epWorldSize", 1))
    tp_ws = int(attrs.get("tpWorldSize", 1))
    shard_type = int(attrs.get("xShardType", 1))
    act_type = int(attrs.get("actType", 0))
    is_bias = bool(attrs.get("isBias", False))
    is_trans = bool(attrs.get("isTrans", False))
    need_ag_out = bool(attrs.get("needAllgatherOut", True))
    need_act_feat = bool(attrs.get("needActivationFeature", False))
    dtype_map = _dtype_map()

    rank_third_parties = None
    try:
        cascade_outs = run_a2a_ag_bmm_cascade(
            thread_contexts,
            device_ids,
            ep_ws=ep_ws,
            tp_ws=tp_ws,
            shard_type=shard_type,
            is_trans=is_trans,
            is_bias=is_bias,
            act_type=act_type,
            need_ag_out=need_ag_out,
            need_act_feat=need_act_feat,
        )
        rank_third_parties = {}
        for did in device_ids:
            tp_list = [cascade_outs[did]["main"]]
            if need_ag_out:
                tp_list.append(cascade_outs[did].get("allgather"))
            if need_act_feat:
                tp_list.append(cascade_outs[did].get("bmm"))
            rank_third_parties[did] = tp_list
        logging.info("AlltoAllAllGatherBatchMatMul: real HCCL cascade succeeded")
    except Exception:
        logging.exception(
            "AlltoAllAllGatherBatchMatMul: real HCCL cascade failed, no third_party"
        )
        rank_third_parties = None

    x_shape = thread_contexts[device_ids[0]].flatten_tensors[0].shape
    E = x_shape[0]
    E_div_ep = E // ep_ws
    if shard_type == 0:
        C = x_shape[1]
        H_div_tp = x_shape[2]
    else:
        C_div_tp = x_shape[1]
        H = x_shape[2]
    n_ep_groups = world_size // ep_ws
    n_tp_groups = world_size // tp_ws
    # EP domain: strided by tp_ws (matching TTK profiling.py HCCL comm creation)
    # EP groups: [[0,4],[1,5],[2,6],[3,7]] for ep=2,tp=4,world=8
    ep_groups = [[x * tp_ws + i for x in range(ep_ws)] for i in range(tp_ws)]
    # TP domain: contiguous (matching TTK profiling.py HCCL comm creation)
    # TP groups: [[0,1,2,3],[4,5,6,7]] for ep=2,tp=4,world=8
    tp_groups = [[x + tp_ws * i for x in range(tp_ws)] for i in range(ep_ws)]
    all_a2a_dids = sorted(set(did for grp in ep_groups for did in grp))
    x_cache = {
        did: to_torch_f32(thread_contexts[did].flatten_tensors[0])
        for did in all_a2a_dids
    }

    a2a_per_rank = {}
    for group_dids in ep_groups:
        chunks_per_rank = {}
        for local_idx, did in enumerate(group_dids):
            chunks_per_rank[local_idx] = x_cache[did].chunk(ep_ws, dim=0)
        for target_local, target_did in enumerate(group_dids):
            result_chunks = [
                chunks_per_rank[src_local][target_local]
                for src_local in range(len(group_dids))
            ]
            a2a_out = torch.cat(result_chunks, dim=0)
            if shard_type == 0:
                a2a_out = (
                    a2a_out.reshape(ep_ws, E_div_ep, C, H_div_tp)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            else:
                a2a_out = (
                    a2a_out.reshape(ep_ws, E_div_ep, C_div_tp, H)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            a2a_per_rank[target_did] = a2a_out
    del x_cache

    for tp_group_dids in tp_groups:
        all_parts = [a2a_per_rank[did] for did in tp_group_dids]
        gathered = torch.cat(all_parts, dim=0)
        del all_parts
        if shard_type == 0:
            gathered = gathered.reshape(tp_ws, E_div_ep, ep_ws, C, H_div_tp)
            gathered = gathered.permute(1, 2, 3, 0, 4).contiguous()
            gathered = gathered.reshape(E_div_ep, ep_ws * C, H_div_tp * tp_ws)
        else:
            gathered = gathered.reshape(tp_ws, E_div_ep, ep_ws, C_div_tp, H)
            gathered = gathered.permute(1, 2, 0, 3, 4).contiguous()
            gathered = gathered.reshape(E_div_ep, ep_ws * tp_ws * C_div_tp, H)

        for did in tp_group_dids:
            tc = thread_contexts[did]
            in_dtype = tc.flatten_tensors[0].dtype
            weight = to_torch_f32(tc.flatten_tensors[1])
            if is_trans:
                weight = weight.permute(0, 2, 1).contiguous()
            # bmm in fp32 then truncate to in_dtype (matching NPU bf16 bmm output)
            bmm_out = torch.bmm(gathered.float(), weight)
            bmm_out = bmm_out.to(in_dtype).float()
            del weight
            if is_bias and len(tc.flatten_tensors) > 2:
                bias = tc.flatten_tensors[2]
                if bias.numel() > 0:
                    bias_f = bias.float()
                    if bias.dim() == 2:
                        bias_f = bias_f.reshape(bias_f.shape[0], 1, bias_f.shape[1])
                    # NPU: bmm(bf16) -> cast fp32 -> + bias(fp32) -> cast bf16
                    bmm_out = (bmm_out.float() + bias_f).to(in_dtype).float()
                del bias
            # activation: compute in fp32 on bf16-truncated input, truncate output to in_dtype
            activated = _apply_act(bmm_out, act_type).to(in_dtype).float()
            bmm_out = bmm_out.to(in_dtype).float()
            goldens = {"main": activated}
            if need_ag_out:
                goldens["allgather"] = gathered.to(in_dtype).float()
            if need_act_feat:
                goldens["bmm"] = bmm_out
            else:
                del bmm_out

            out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
            golden_list = []
            out_keys = ["main", "allgather", "bmm"]
            for oi, out_idx in enumerate(tc.output_tensor_indexes):
                if oi < len(out_keys):
                    g = goldens.get(out_keys[oi])
                else:
                    g = None
                if g is None:
                    g = torch.zeros(tc.tensor_view_shapes[out_idx])
                dt_idx = list(tc.output_tensor_indexes).index(out_idx)
                if dt_idx < len(out_dtypes):
                    target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                    if target_dtype is not None:
                        g = g.to(target_dtype)
                golden_list.append(g.contiguous())
            tc.golden_tensors = golden_list
            third_parties_list = None
            if rank_third_parties is not None:
                tp_items = rank_third_parties.get(did)
                if tp_items is not None:
                    third_parties_list = []
                    for oi_idx, out_idx in enumerate(tc.output_tensor_indexes):
                        if oi_idx < len(tp_items):
                            tp = tp_items[oi_idx]
                            if isinstance(tp, torch.Tensor):
                                out_shape = tc.tensor_view_shapes[out_idx]
                                if tp.shape != torch.Size(out_shape):
                                    tp = tp.reshape(out_shape)
                                dt_idx = list(tc.output_tensor_indexes).index(out_idx)
                                if dt_idx < len(out_dtypes):
                                    target_dtype = dtype_map.get(
                                        out_dtypes[dt_idx], None
                                    )
                                    if target_dtype is not None:
                                        tp = tp.to(target_dtype)
                                third_parties_list.append(tp.contiguous())
                            else:
                                third_parties_list.append(None)
                        else:
                            third_parties_list.append(None)
            try:
                cr = Comparator(tc).compare(third_parties=third_parties_list)
                msg = f"rank{did}:{cr.passed}({fmt_compare(cr)})"
                if cr.passed != "PASS":
                    logging.error(f"A2A_AG_BMM: {msg}")
                else:
                    logging.info(f"A2A_AG_BMM: {msg}")
                all_precision.append(msg)
            except Exception:
                logging.exception(f"A2A_AG_BMM: rank dev={did} comparison failure")
                all_precision.append(f"rank{did}:COMPARE_EXCEPTION")
            del goldens, activated
        del gathered
    del a2a_per_rank


def a2a_ag_bmm_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    x_cpu = cpu_inputs_per_rank[rank][0].float()
    w_cpu = cpu_inputs_per_rank[rank][1].float()
    shard_type = int(attrs.get("shard_type", attrs.get("xShardType", 0)))
    ep_ws = int(attrs.get("epWorldSize", 0))
    tp_ws = int(attrs.get("tpWorldSize", 0))
    act_type_val = attrs.get("actType", 0)
    if isinstance(act_type_val, str):
        act_map = {"none": 0, "gelu": 1, "silu": 2, "relu": 3, "fastgelu": 4}
        act_type = act_map.get(act_type_val.lower(), 0)
    else:
        act_type = int(act_type_val)
    is_bias = bool(attrs.get("isBias", False))
    need_ag_out = bool(
        attrs.get("needAllgatherOut", attrs.get("need_allgather_out", False))
    )
    need_act_feat = bool(
        attrs.get("needActivationFeature", attrs.get("need_activation_feature", False))
    )
    if not ep_ws or not tp_ws:
        return None

    E_local = x_cpu.shape[0]
    E_div_ep = E_local // ep_ws
    if shard_type == 0:
        C = x_cpu.shape[1]
        H_div_tp = x_cpu.shape[2]
    else:
        C_div_tp = x_cpu.shape[1]
        H = x_cpu.shape[2]

    n_ep_groups = ws // ep_ws
    n_tp_groups = ws // tp_ws
    # EP domain: strided by tp_ws (matching TTK profiling.py HCCL comm creation)
    ep_groups = [[x * tp_ws + i for x in range(ep_ws)] for i in range(tp_ws)]
    # TP domain: contiguous (matching TTK profiling.py HCCL comm creation)
    tp_groups = [[x + tp_ws * i for x in range(tp_ws)] for i in range(ep_ws)]
    all_x_cache = {}
    for r in range(ws):
        all_x_cache[r] = cpu_inputs_per_rank[r][0].float()

    a2a_per_rank = {}
    for group_ranks in ep_groups:
        chunks_per_rank = {}
        for local_idx, r in enumerate(group_ranks):
            chunks_per_rank[local_idx] = all_x_cache[r].chunk(ep_ws, dim=0)
        for target_local, target_rank in enumerate(group_ranks):
            result_chunks = [
                chunks_per_rank[src_local][target_local]
                for src_local in range(len(group_ranks))
            ]
            a2a_out = torch.cat(result_chunks, dim=0)
            if shard_type == 0:
                a2a_out = (
                    a2a_out.reshape(ep_ws, E_div_ep, C, H_div_tp)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            else:
                a2a_out = (
                    a2a_out.reshape(ep_ws, E_div_ep, C_div_tp, H)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )
            a2a_per_rank[target_rank] = a2a_out

    for tp_group_ranks in tp_groups:
        all_parts = [a2a_per_rank[r] for r in tp_group_ranks]
        gathered = torch.cat(all_parts, dim=0)
        del all_parts
        if shard_type == 0:
            gathered = gathered.reshape(tp_ws, E_div_ep, ep_ws, C, H_div_tp)
            gathered = gathered.permute(1, 2, 3, 0, 4).contiguous()
            gathered = gathered.reshape(E_div_ep, ep_ws * C, H_div_tp * tp_ws)
        else:
            gathered = gathered.reshape(tp_ws, E_div_ep, ep_ws, C_div_tp, H)
            gathered = gathered.permute(1, 2, 0, 3, 4).contiguous()
            gathered = gathered.reshape(E_div_ep, ep_ws * tp_ws * C_div_tp, H)

        for r in tp_group_ranks:
            if r == rank:
                w_r = cpu_inputs_per_rank[r][1].float()
                bmm_out = torch.bmm(gathered, w_r)
                del w_r
                if is_bias and len(cpu_inputs_per_rank[r]) > 2:
                    bias_r = cpu_inputs_per_rank[r][2].float()
                    if bias_r.numel() > 0:
                        if bias_r.dim() == 2:
                            bias_r = bias_r.reshape(bias_r.shape[0], 1, bias_r.shape[1])
                        bmm_out = bmm_out + bias_r
                    del bias_r
                activated = _apply_act(bmm_out, act_type)
                in_dtype = cpu_inputs_per_rank[r][0].dtype
                goldens = {"main": activated.to(in_dtype).float()}
                if need_ag_out:
                    goldens["allgather"] = gathered.to(in_dtype).float()
                if need_act_feat:
                    goldens["bmm"] = bmm_out.to(in_dtype).float()
                return [goldens]
        del gathered
    return None
