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

"""Golden for aclnnAlltoAllvGroupedMatMul / torch_npu.npu_alltoallv_gmm.

AlltoAllvGroupedMatMul: all_to_allv(x) -> permute -> npu_gmm -> output [+ mm_out]
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
    return mod.to_torch_f32, mod.apply_gmm_goldens


# ---------------------------------------------------------------------------
# shared GMM helpers (self-contained, no _common.py)
# ---------------------------------------------------------------------------


def _grouped_matmul_cpu(gmm_x, gmm_weight, group_list):
    import torch

    B_list = list(torch.unbind(gmm_weight, dim=0))
    A_groups = torch.split(gmm_x, group_list, dim=0)
    results = []
    for i in range(len(group_list)):
        results.append(
            torch.from_numpy(np.matmul(A_groups[i].numpy(), B_list[i].numpy()))
        )
    return torch.cat(results, dim=0)


def _generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed):
    n = len(A_array)
    rng = np.random.default_rng(seed)
    total = sum(A_array)
    if total % n != 0:
        return [[total // n] * (exp_per_card * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in A_array:
        if a % n != 0:
            return [[col_sum // exp_per_card] * (exp_per_card * n) for _ in range(n)]
        k = a // n
        k_values.append(max(k, exp_per_card))
    blocks = []
    for k in k_values:
        block = np.zeros((exp_per_card, n), dtype=int)
        for col in range(n):
            counts = rng.multinomial(
                k - exp_per_card, [1.0 / exp_per_card] * exp_per_card
            )
            block[:, col] = counts + 1
        blocks.append(block)
    tmp = np.vstack(blocks)
    return [list(col) for col in zip(*tmp)]


def _get_gmm_exp_token_nums(first_ctx, ep_ws):
    exp_per_card = (
        first_ctx.tensor_view_shapes[1][0]
        if len(first_ctx.tensor_view_shapes) > 1
        else 1
    )
    seed_val = 0
    remark = first_ctx.remark or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "seed":
            try:
                seed_val = int(kv[1].strip())
            except ValueError:
                pass
    bsk = first_ctx.tensor_view_shapes[0][0] if first_ctx.tensor_view_shapes else 0
    A_array = [bsk] * ep_ws
    return _generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)


def _get_gmm_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws):
    return [
        sum(expTokenNums[i][rank_idx * exp_per_card + j] for i in range(ep_ws))
        for j in range(exp_per_card)
    ]


def _permute_a2a_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch

    indices = np.zeros((exp_per_card, ep_ws), dtype=np.int64)
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = int(expTokenNums[i][j + exp_per_card * rank_idx])
    trans = indices.T
    flaten = trans.reshape(-1)
    cumsum = np.cumsum(flaten)
    all_indices = []
    for e in range(exp_per_card):
        exp_token = []
        for r in range(ep_ws):
            flat_idx = e + r * exp_per_card
            start = int(cumsum[flat_idx - 1]) if flat_idx > 0 else 0
            end = int(cumsum[flat_idx])
            exp_token.extend(range(start, end))
        all_indices.extend(exp_token)
    if len(all_indices) == 0:
        return tokens.clone(), []
    idx_tensor = torch.tensor(all_indices, dtype=torch.long)
    expert_sizes = [
        sum(expTokenNums[i][rank_idx * exp_per_card + e] for i in range(ep_ws))
        for e in range(exp_per_card)
    ]
    return tokens.index_select(0, idx_tensor), expert_sizes


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


_next_port_base = [30060]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30060
        return port


def run_alltoallv_gmm_cascade(
    thread_contexts,
    device_ids,
    expTokenNums,
    ep_ws,
    exp_per_card,
    trans_gmm_weight=False,
    trans_mm_weight=False,
    permute_out_flag=False,
    mm_out_flag=False,
):
    import subprocess
    import sys
    import torch

    n = len(device_ids)
    if n < 2:
        return {}
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_a2agmm_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)
        np.savez(
            input_path + ".meta.npz",
            expTokenNums=np.array(expTokenNums, dtype=np.int64),
            ep_ws=np.array([ep_ws], dtype=np.int64),
            exp_per_card=np.array([exp_per_card], dtype=np.int64),
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
                str(trans_gmm_weight),
                str(trans_mm_weight),
                str(permute_out_flag),
                str(mm_out_flag),
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
        gmm_weight = first_ctx.flatten_tensors[1]
        n_dim = gmm_weight.shape[1] if trans_gmm_weight else gmm_weight.shape[2]
        K = first_ctx.flatten_tensors[0].shape[1]
        result = {}
        for did in device_ids:
            rank_idx = list(device_ids).index(did)
            recv_total = sum(
                sum(
                    expTokenNums[i][
                        rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card
                    ]
                )
                for i in range(ep_ws)
            )
            main_t = None
            rank_file = f"{result_path}.did{did}.npz"
            if os.path.exists(rank_file):
                data = np.load(rank_file, allow_pickle=False)
                key = f"cascade_did{did}"
                if key in data.files:
                    main_t = torch.from_numpy(data[key].copy()).reshape(
                        recv_total, n_dim
                    )
            permute_t = None
            if permute_out_flag:
                perm_file = f"{result_path}.a2a_did{did}.npz"
                if os.path.exists(perm_file):
                    data = np.load(perm_file, allow_pickle=False)
                    key = f"cascade_a2a_did{did}"
                    if key in data.files:
                        permute_t = torch.from_numpy(data[key].copy()).reshape(
                            recv_total, K
                        )
            mm_t = None
            mm_file = f"{result_path}.mm_did{did}.npz"
            if os.path.exists(mm_file):
                data = np.load(mm_file, allow_pickle=False)
                key = f"cascade_mm_did{did}"
                if key in data.files:
                    mm_t = torch.from_numpy(data[key].copy())
            result[did] = {"main": main_t, "permute": permute_t, "mm": mm_t}
        return result


# ---------------------------------------------------------------------------
# CPU golden
# ---------------------------------------------------------------------------


def alltoallv_gmm_cpu_golden(
    thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card
):
    import torch

    to_torch_f32, _ = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    trans_gmm_weight = attrs.get("transGmmWeight", False)
    trans_mm_weight = attrs.get("transMmWeight", False)
    permute_out_flag = attrs.get("permuteOutFlag", False)

    all_a2a_inputs = {}
    all_send_segments = {}
    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        src_x = to_torch_f32(tc.flatten_tensors[0])
        all_a2a_inputs[did] = src_x
        my_row = expTokenNums[rank_idx]
        segments = []
        offset = 0
        for t in range(ep_ws):
            cs = sum(my_row[t * exp_per_card : (t + 1) * exp_per_card])
            segments.append(src_x[offset : offset + cs])
            offset += cs
        all_send_segments[did] = segments

    a2a_outputs = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        output_splits = [
            sum(
                expTokenNums[i][
                    target_idx * exp_per_card : (target_idx + 1) * exp_per_card
                ]
            )
            for i in range(ep_ws)
        ]
        recv_offsets = [0] + list(np.cumsum(output_splits)[:-1])
        K = (
            all_a2a_inputs[device_ids[0]].shape[1]
            if all_a2a_inputs[device_ids[0]].dim() > 1
            else 1
        )
        gathered = torch.zeros(sum(output_splits), K)
        for src_did in device_ids:
            src_idx = list(device_ids).index(src_did)
            chunk = all_send_segments[src_did][target_idx]
            base = recv_offsets[src_idx]
            gathered[base : base + chunk.shape[0]] = chunk
        a2a_outputs[target_did] = gathered

    rank_goldens = {}
    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        gmm_weight = to_torch_f32(tc.flatten_tensors[1])
        if trans_gmm_weight:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
        a2a_out = a2a_outputs[did]
        permuted, expert_sizes = _permute_a2a_gmm(
            a2a_out, exp_per_card, ep_ws, rank_idx, expTokenNums
        )
        gmm_out = _grouped_matmul_cpu(permuted, gmm_weight, expert_sizes)
        rank_goldens[did] = {
            "main": gmm_out.contiguous(),
            "permute": permuted.contiguous() if permute_out_flag else None,
        }
        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_weight_f = to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            rank_goldens[did]["mm"] = torch.mm(to_torch_f32(mm_x), mm_weight_f)
        else:
            rank_goldens[did]["mm"] = None
        del gmm_out, permuted
    del all_a2a_inputs, all_send_segments, a2a_outputs
    return rank_goldens


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


def alltoallv_gmm_multi_device_golden(thread_contexts, device_ids, all_precision):
    import torch

    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    ep_ws = attrs.get("epWorldSize", len(device_ids))
    exp_per_card = (
        first_ctx.tensor_view_shapes[1][0]
        if len(first_ctx.tensor_view_shapes) > 1
        else 1
    )
    expTokenNums = _get_gmm_exp_token_nums(first_ctx, ep_ws)
    trans_gmm_weight = bool(attrs.get("transGmmWeight", False))
    trans_mm_weight = bool(attrs.get("transMmWeight", False))
    permute_out_flag = bool(attrs.get("permuteOutFlag", False))

    rank_goldens = alltoallv_gmm_cpu_golden(
        thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card
    )
    mm_out_flag = rank_goldens.get(device_ids[0], {}).get("mm") is not None

    rank_third_parties = None
    try:
        cascade_outs = run_alltoallv_gmm_cascade(
            thread_contexts,
            device_ids,
            expTokenNums,
            ep_ws,
            exp_per_card,
            trans_gmm_weight=trans_gmm_weight,
            trans_mm_weight=trans_mm_weight,
            permute_out_flag=permute_out_flag,
            mm_out_flag=mm_out_flag,
        )
        rank_third_parties = {}
        for did in device_ids:
            tp_list = [cascade_outs[did]["main"]]
            out_idxs = thread_contexts[did].output_tensor_indexes
            for oi in range(1, len(out_idxs)):
                if oi == 1:
                    if cascade_outs[did].get("mm") is not None:
                        tp_list.append(cascade_outs[did]["mm"])
                    elif cascade_outs[did].get("permute") is not None:
                        tp_list.append(cascade_outs[did]["permute"])
                    else:
                        tp_list.append(None)
                elif oi == 2:
                    tp_list.append(cascade_outs[did].get("permute"))
                else:
                    tp_list.append(None)
            rank_third_parties[did] = tp_list
        logging.info("AlltoAllvGroupedMatMul: real HCCL cascade succeeded")
    except Exception:
        logging.exception(
            "AlltoAllvGroupedMatMul: real HCCL cascade failed, no third_party"
        )
        rank_third_parties = None

    _, apply_gmm = _load_framework()
    apply_gmm(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


def alltoallv_gmm_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import numpy as _np

    epc = int(attrs.get("expPerCard", 1))
    seed = int(attrs.get("seed", 1))
    ep_ws = int(attrs.get("ep_ws", ws))
    expTokenNums = attrs.get("expTokenNums")
    if expTokenNums is None:
        A = cpu_inputs_per_rank[0][0].float().shape[0]
        expTokenNums = _generate_gmm_alltoallv_matrix([A] * ep_ws, epc, seed=seed)
    trans_gmm = bool(attrs.get("transGmmWeight", attrs.get("trans_gmm_weight", False)))
    trans_mm = bool(attrs.get("transMmWeight", attrs.get("trans_mm_weight", False)))
    permute_out_flag = bool(
        attrs.get("permuteOutFlag", attrs.get("permute_out_flag", False))
    )

    all_a2a_inputs = {}
    all_send_segments = {}
    for r in range(ws):
        src_x = cpu_inputs_per_rank[r][0].float()
        all_a2a_inputs[r] = src_x
        my_row = expTokenNums[r]
        segments = []
        offset = 0
        for t in range(ep_ws):
            cs = sum(my_row[t * epc : (t + 1) * epc])
            segments.append(src_x[offset : offset + cs])
            offset += cs
        all_send_segments[r] = segments

    output_splits = [
        sum(expTokenNums[i][rank * epc : (rank + 1) * epc]) for i in range(ep_ws)
    ]
    recv_offsets = [0] + list(_np.cumsum(output_splits)[:-1])
    K = all_a2a_inputs[0].shape[1] if all_a2a_inputs[0].dim() > 1 else 1
    gathered = torch.zeros(sum(output_splits), K, dtype=torch.float32)
    for src_r in range(ws):
        chunk = all_send_segments[src_r][rank]
        gathered[recv_offsets[src_r] : recv_offsets[src_r] + chunk.shape[0]] = chunk

    permuted, expert_sizes = _permute_a2a_gmm(gathered, epc, ep_ws, rank, expTokenNums)
    gmm_weight = cpu_inputs_per_rank[rank][1].float()
    if trans_gmm:
        gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
    main_golden = _grouped_matmul_cpu(permuted, gmm_weight, expert_sizes).float()
    permute_ret = permuted.contiguous() if permute_out_flag else None

    mm_golden = None
    if len(cpu_inputs_per_rank[rank]) > 2 and cpu_inputs_per_rank[rank][2] is not None:
        mm_x = cpu_inputs_per_rank[rank][2].float()
        mm_weight = cpu_inputs_per_rank[rank][3].float()
        if trans_mm:
            mm_weight = mm_weight.t().contiguous()
        mm_golden = torch.mm(mm_x, mm_weight)
    del all_a2a_inputs, all_send_segments
    if permute_ret is not None:
        return (main_golden, mm_golden, permute_ret)
    if mm_golden is not None:
        return (main_golden, mm_golden)
    return main_golden
