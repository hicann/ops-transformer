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

"""Golden for aclnnGroupedMatMulAlltoAllv / torch_npu.npu_gmm_alltoallv.

GroupedMatMulAlltoAllv: npu_gmm(x, weight) -> unpermute -> all_to_allv -> output
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
# shared GMM helpers
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


def _unpermute_mc2(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch

    empty_arr = np.zeros((ep_ws, exp_per_card), dtype=np.int64)
    for i in range(ep_ws):
        for j in range(exp_per_card):
            empty_arr[i][j] = int(expTokenNums[i][rank_idx * exp_per_card + j])
    tmp1 = empty_arr.T
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2 = np.cumsum(sum_list1)
    offsets = [0] + sum_list2[:-1].tolist()
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for ei in range(exp_per_card):
        tmp = []
        for j in range(ep_ws):
            if j == 0:
                tmp.append(
                    list(
                        map(lambda x: x + offsets[ei], list(range(0, sum_list[ei][j])))
                    )
                )
            else:
                tmp.append(
                    list(
                        map(
                            lambda x: x + offsets[ei],
                            list(range(sum_list[ei][j - 1], sum_list[ei][j])),
                        )
                    )
                )
        indices_list.append(tmp)
    selected = []
    for i in range(ep_ws):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long)
            selected.append(tokens.index_select(dim=0, index=indices))
    return torch.cat(selected, dim=0).to(tokens.dtype)


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


_next_port_base = [30052]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30052
        return port


def run_gmm_alltoallv_cascade(
    thread_contexts,
    device_ids,
    expTokenNums,
    ep_ws,
    exp_per_card,
    trans_gmm_weight=False,
    trans_mm_weight=False,
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
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_gmma2a_") as tmpdir:
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
            mm_t = None
            mm_file = f"{result_path}.mm_did{did}.npz"
            if os.path.exists(mm_file):
                data = np.load(mm_file, allow_pickle=False)
                key = f"cascade_mm_did{did}"
                if key in data.files:
                    mm_t = torch.from_numpy(data[key].copy())
            result[did] = {"main": main_t, "mm": mm_t}
        return result


# ---------------------------------------------------------------------------
# CPU golden
# ---------------------------------------------------------------------------


def gmm_alltoallv_cpu_golden(
    thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card
):
    import torch

    to_torch_f32, _ = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    trans_gmm_weight = attrs.get("transGmmWeight", False)
    trans_mm_weight = attrs.get("transMmWeight", False)

    all_gmm_out = {}
    all_unpermuted = {}
    for did in device_ids:
        tc = thread_contexts[did]
        rank_idx = list(device_ids).index(did)
        gmm_x = to_torch_f32(tc.flatten_tensors[0])
        gmm_weight = to_torch_f32(tc.flatten_tensors[1])
        if trans_gmm_weight:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
        recv_gl = _get_gmm_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws)
        gmm_out = _grouped_matmul_cpu(gmm_x, gmm_weight, recv_gl)
        all_gmm_out[did] = gmm_out
        all_unpermuted[did] = _unpermute_mc2(
            gmm_out, exp_per_card, ep_ws, rank_idx, expTokenNums
        )

    rank_goldens = {}
    for target_did in device_ids:
        tc = thread_contexts[target_did]
        target_rank = list(device_ids).index(target_did)
        N = (
            all_unpermuted[device_ids[0]].shape[1]
            if all_unpermuted[device_ids[0]].dim() > 1
            else 1
        )
        output_chunks = []
        for src_did in device_ids:
            src_rank = list(device_ids).index(src_did)
            src_unpermuted = all_unpermuted[src_did]
            is_list = [
                sum(expTokenNums[src_rank][t * exp_per_card : (t + 1) * exp_per_card])
                for t in range(ep_ws)
            ]
            offset = 0
            for t in range(ep_ws):
                if t == target_rank:
                    output_chunks.append(
                        src_unpermuted[offset : offset + is_list[t]].clone()
                    )
                offset += is_list[t]
        main_golden = (
            torch.cat(output_chunks, dim=0) if output_chunks else torch.zeros(0, N)
        )
        rank_goldens[target_did] = {"main": main_golden}
        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_weight_f = to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            rank_goldens[target_did]["mm"] = torch.mm(to_torch_f32(mm_x), mm_weight_f)
        else:
            rank_goldens[target_did]["mm"] = None
        del output_chunks, main_golden
    del all_gmm_out, all_unpermuted
    return rank_goldens


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


def gmm_alltoallv_multi_device_golden(thread_contexts, device_ids, all_precision):
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

    rank_goldens = gmm_alltoallv_cpu_golden(
        thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card
    )
    mm_out_flag = rank_goldens.get(device_ids[0], {}).get("mm") is not None

    rank_third_parties = None
    try:
        cascade_outs = run_gmm_alltoallv_cascade(
            thread_contexts,
            device_ids,
            expTokenNums,
            ep_ws,
            exp_per_card,
            trans_gmm_weight=trans_gmm_weight,
            trans_mm_weight=trans_mm_weight,
            mm_out_flag=mm_out_flag,
        )
        rank_third_parties = {}
        for did in device_ids:
            tp_list = [cascade_outs[did]["main"]]
            out_idxs = thread_contexts[did].output_tensor_indexes
            for oi in range(1, len(out_idxs)):
                if oi == 1:
                    tp_list.append(cascade_outs[did].get("mm"))
                else:
                    tp_list.append(None)
            rank_third_parties[did] = tp_list
        logging.info("GroupedMatMulAlltoAllv: real HCCL cascade succeeded")
    except Exception:
        logging.exception(
            "GroupedMatMulAlltoAllv: real HCCL cascade failed, no third_party"
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


def gmm_alltoallv_e2e_golden(
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

    all_gmm_out = {}
    all_unpermuted = {}
    for r in range(ws):
        gmm_x = cpu_inputs_per_rank[r][0].float()
        gmm_weight = cpu_inputs_per_rank[r][1].float()
        if trans_gmm:
            gmm_weight = gmm_weight.permute(0, 2, 1).contiguous()
        recv_gl = _get_gmm_group_list(expTokenNums, r, epc, ep_ws)
        gmm_out = _grouped_matmul_cpu(gmm_x, gmm_weight, recv_gl)
        all_gmm_out[r] = gmm_out
        all_unpermuted[r] = _unpermute_mc2(gmm_out, epc, ep_ws, r, expTokenNums)

    target_rank = rank
    output_chunks = []
    for src_r in range(ws):
        src_unpermuted = all_unpermuted[src_r]
        is_list = [
            sum(expTokenNums[src_r][t * epc : (t + 1) * epc]) for t in range(ep_ws)
        ]
        offset = 0
        for t in range(ep_ws):
            if t == target_rank:
                output_chunks.append(
                    src_unpermuted[offset : offset + is_list[t]].clone()
                )
            offset += is_list[t]
    N = all_unpermuted[0].shape[1] if all_unpermuted[0].dim() > 1 else 1
    main_golden = (
        torch.cat(output_chunks, dim=0) if output_chunks else torch.zeros(0, N)
    )
    mm_golden = None
    if len(cpu_inputs_per_rank[rank]) > 2 and cpu_inputs_per_rank[rank][2] is not None:
        mm_x = cpu_inputs_per_rank[rank][2].float()
        mm_weight = cpu_inputs_per_rank[rank][3].float()
        if trans_mm:
            mm_weight = mm_weight.t().contiguous()
        mm_golden = torch.mm(mm_x, mm_weight)
    del all_gmm_out, all_unpermuted
    if mm_golden is not None:
        return (main_golden, mm_golden)
    return main_golden
