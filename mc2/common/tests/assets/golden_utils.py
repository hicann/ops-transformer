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

"""Shared golden utilities for MC2 operators.

Extracted from TTK's mc2_golden.py so that per-operator goldens in
ops-transformer are self-contained. TTK retains only the multi-device
test framework (process spawning, HCCL env, device management).
"""

import logging

import numpy


# ---------------------------------------------------------------------------
# Dtype conversion helpers
# ---------------------------------------------------------------------------


def to_torch_f32(t):
    """Convert torch.Tensor or numpy.ndarray to float32 torch.Tensor."""
    import torch

    if t is None:
        return None
    if isinstance(t, numpy.ndarray):
        dtype_str = str(t.dtype)
        if "e8m0" in dtype_str or "float8_e8m0" in dtype_str:
            raw = t.view(numpy.uint8).astype(numpy.float64)
            arr = numpy.power(2.0, raw - 127).astype(numpy.float32)
        else:
            try:
                arr = t.astype(numpy.float32, copy=False)
            except (TypeError, ValueError):
                arr = t.view(numpy.uint8).astype(numpy.float32, copy=False)
        return torch.from_numpy(arr)
    if hasattr(t, "dtype") and str(t.dtype).replace("torch.", "") in (
        "float8_e4m3fn",
        "float8_e5m2",
        "hifloat8",
    ):
        return t.float()
    return t.float()


def e8m0_to_f32(scale_tensor):
    """Convert e8m0 scales to float32 using 2^(e-127) formula."""
    import torch

    if scale_tensor is None:
        return None
    if isinstance(scale_tensor, numpy.ndarray):
        raw = scale_tensor.view(numpy.uint8).astype(numpy.float64)
        return numpy.power(2.0, raw - 127).astype(numpy.float32)
    dtype_str = str(scale_tensor.dtype).replace("torch.", "")
    if "e8m0" in dtype_str:
        raw = scale_tensor.view(torch.uint8).to(torch.float64)
        return torch.pow(2.0, raw - 127).to(torch.float32)
    return scale_tensor.float()


def to_torch_keep(t):
    """Convert numpy.ndarray to torch.Tensor, keep dtype if torch.Tensor."""
    import torch

    if t is None:
        return None
    if isinstance(t, numpy.ndarray):
        return torch.from_numpy(t)
    return t


# ---------------------------------------------------------------------------
# Comparison framework helpers
# ---------------------------------------------------------------------------


def fmt_compare_result(cr):
    """Format compare result with mere/mare for stat_rel_err or precision for close."""
    _m = cr.metrics.get(0, {}) if cr.metrics else {}
    if _m and "mere" in _m:
        return f" mere={_m.get('mere'):.4e} mare={_m.get('mare'):.4e} th={_m.get('threshold'):.4e}"
    return f" prec={cr.precision}"


def _get_comparator():
    from ttk.core_modules.npu.op_api.comparison import Comparator

    return Comparator


def apply_goldens_and_compare(
    thread_contexts, device_ids, rank_goldens, all_precision, rank_third_parties=None
):
    """Apply golden tensors to each rank's context and run Comparator.

    rank_goldens: dict did -> tensor OR did -> {'main':.., 'gather':..}
    """
    import torch as _torch
    import numpy as _np

    Comparator = _get_comparator()
    dtype_map = {
        "float16": _torch.float16,
        "fp16": _torch.float16,
        "float32": _torch.float32,
        "fp32": _torch.float32,
        "bfloat16": _torch.bfloat16,
        "bf16": _torch.bfloat16,
    }
    torch_native_dtypes = {
        "float16",
        "fp16",
        "float32",
        "fp32",
        "bfloat16",
        "bf16",
        "int8",
        "int32",
        "int64",
        "uint8",
        "bool",
    }
    for did in device_ids:
        tc = thread_contexts[did]
        golden = rank_goldens[did]
        multi_output_goldens = None
        if isinstance(golden, dict):
            multi_output_goldens = [golden.get("main"), golden.get("gather")]
            golden = multi_output_goldens[0]
        third_party = rank_third_parties[did] if rank_third_parties else None
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        use_torch = tc.is_torch_dtype_support()
        target_dtype = None
        if len(out_dtypes) > 0:
            out_dtype_str = out_dtypes[0]
            target_dtype = dtype_map.get(out_dtype_str, None)
            if target_dtype is not None and use_torch:
                golden = golden.to(target_dtype)
            else:
                golden_t = (
                    golden.float()
                    if hasattr(golden, "float")
                    else _torch.from_numpy(_np.asarray(golden).astype(_np.float32))
                )
                if out_dtype_str in ("bfloat16", "bf16"):
                    golden_t = golden_t.to(_torch.bfloat16).float()
                elif out_dtype_str in ("float16", "fp16"):
                    golden_t = golden_t.to(_torch.float16).float()
                golden = golden_t.numpy().astype(_np.float32, copy=False)
        elif not use_torch:
            golden = golden.float().numpy().astype(_np.float32, copy=False)
        if multi_output_goldens is None:
            tc.golden_tensors = [
                golden.contiguous() if isinstance(golden, _torch.Tensor) else golden
            ]
        else:
            converted_goldens = []
            for output_position, output_golden in enumerate(multi_output_goldens):
                if output_golden is None:
                    output_golden = _torch.zeros(
                        tc.tensor_view_shapes[tc.output_tensor_indexes[output_position]]
                    )
                if output_position < len(out_dtypes):
                    output_dtype = dtype_map.get(out_dtypes[output_position], None)
                    if output_dtype is not None and use_torch:
                        output_golden = output_golden.to(output_dtype)
                converted_goldens.append(output_golden.contiguous())
            tc.golden_tensors = converted_goldens[: len(tc.output_tensor_indexes)]
        third_parties_list = None
        if third_party is not None:
            if isinstance(third_party, (list, tuple)):
                tp_items = list(third_party)
            else:
                tp_items = [third_party]
            if target_dtype is not None and use_torch:
                tp_items = [
                    tp.to(target_dtype) if hasattr(tp, "to") else tp for tp in tp_items
                ]
            third_parties_list = [
                tp.contiguous() if isinstance(tp, _torch.Tensor) else tp
                for tp in tp_items
            ]
        del rank_goldens[did]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"Multi-device: rank dev={did} comparison FAILED: {cr.precision} metrics={cr.metrics}"
                )
            else:
                logging.info(f"Multi-device: rank dev={did} comparison PASSED")
        except Exception:
            logging.exception(f"Multi-device: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def apply_a2a_goldens_and_compare(
    thread_contexts, device_ids, rank_goldens, all_precision, rank_third_parties=None
):
    """Apply golden tensors for AlltoAll-type ops (main + alltoall outputs)."""
    import torch as _torch

    Comparator = _get_comparator()
    dtype_map = {
        "float16": _torch.float16,
        "fp16": _torch.float16,
        "float32": _torch.float32,
        "fp32": _torch.float32,
        "bfloat16": _torch.bfloat16,
        "bf16": _torch.bfloat16,
    }
    for did in device_ids:
        tc = thread_contexts[did]
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        goldens = rank_goldens[did]
        third_party = rank_third_parties[did] if rank_third_parties else None
        golden_list = []
        third_parties_list = None
        for out_idx in tc.output_tensor_indexes:
            if isinstance(goldens, dict):
                if out_idx == tc.output_tensor_indexes[0]:
                    g = goldens["main"]
                elif (
                    len(tc.output_tensor_indexes) > 1
                    and out_idx == tc.output_tensor_indexes[1]
                ):
                    g = goldens.get("alltoall")
                else:
                    g = _torch.zeros(tc.tensor_view_shapes[out_idx])
                if g is None:
                    g = _torch.zeros(tc.tensor_view_shapes[out_idx])
            else:
                g = goldens
            dt_idx = list(tc.output_tensor_indexes).index(out_idx)
            if dt_idx < len(out_dtypes):
                target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                if target_dtype is not None:
                    g = g.to(target_dtype)
            golden_list.append(g.contiguous())
        if third_party is not None:
            tp_items = (
                third_party if isinstance(third_party, (list, tuple)) else [third_party]
            )
            third_parties_list = []
            for oi_idx, out_idx in enumerate(tc.output_tensor_indexes):
                if oi_idx < len(tp_items):
                    tp = tp_items[oi_idx]
                    third_parties_list.append(
                        tp.contiguous() if isinstance(tp, _torch.Tensor) else tp
                    )
                else:
                    third_parties_list.append(None)
        tc.golden_tensors = golden_list
        del rank_goldens[did]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"Multi-device: rank dev={did} comparison FAILED: {cr.precision} metrics={cr.metrics}"
                )
            else:
                logging.info(f"Multi-device: rank dev={did} comparison PASSED")
        except Exception:
            logging.exception(f"Multi-device: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


def apply_gmm_goldens(
    thread_contexts, device_ids, rank_goldens, all_precision, rank_third_parties=None
):
    """Apply golden tensors for GMM-type ops (main + mm + permute outputs)."""
    import torch as _torch

    Comparator = _get_comparator()
    dtype_map = {
        "float16": _torch.float16,
        "fp16": _torch.float16,
        "float32": _torch.float32,
        "fp32": _torch.float32,
        "bfloat16": _torch.bfloat16,
        "bf16": _torch.bfloat16,
    }
    for did in device_ids:
        tc = thread_contexts[did]
        out_dtypes = tc.flat_output_dtypes if tc.flat_output_dtypes else []
        goldens = rank_goldens[did]
        third_party = rank_third_parties[did] if rank_third_parties else None
        golden_list = []
        third_parties_list = None
        for out_idx in tc.output_tensor_indexes:
            shape = tc.tensor_view_shapes[out_idx]
            if shape is None or any(
                s is None for s in shape if isinstance(shape, (list, tuple))
            ):
                golden_list.append(None)
                continue
            if out_idx == tc.output_tensor_indexes[0]:
                g = goldens["main"]
            elif (
                len(tc.output_tensor_indexes) > 1
                and out_idx == tc.output_tensor_indexes[1]
            ):
                g = goldens.get("mm")
                if g is None:
                    g = goldens.get("permute")
            elif (
                len(tc.output_tensor_indexes) > 2
                and out_idx == tc.output_tensor_indexes[2]
            ):
                g = goldens.get("permute")
            else:
                g = _torch.zeros(shape)
            if g is None:
                g = _torch.zeros(shape)
            dt_idx = list(tc.output_tensor_indexes).index(out_idx)
            if dt_idx < len(out_dtypes):
                target_dtype = dtype_map.get(out_dtypes[dt_idx], None)
                if target_dtype is not None:
                    g = g.to(target_dtype)
            if g.shape != _torch.Size(shape):
                g = g.reshape(shape)
            golden_list.append(g.contiguous())
        if third_party is not None:
            tp_items = (
                third_party if isinstance(third_party, (list, tuple)) else [third_party]
            )
            third_parties_list = []
            for oi_idx, out_idx in enumerate(tc.output_tensor_indexes):
                if oi_idx < len(tp_items):
                    tp = tp_items[oi_idx]
                    third_parties_list.append(
                        tp.contiguous() if isinstance(tp, _torch.Tensor) else tp
                    )
                else:
                    third_parties_list.append(None)
        tc.golden_tensors = golden_list
        del rank_goldens[did]
        try:
            cr = Comparator(tc).compare(third_parties=third_parties_list)
            all_precision.append(f"rank{did}:{cr.passed}({fmt_compare_result(cr)})")
            if cr.passed != "PASS":
                logging.error(
                    f"Multi-device: rank dev={did} comparison FAILED: {cr.precision} metrics={cr.metrics}"
                )
            else:
                logging.info(f"Multi-device: rank dev={did} comparison PASSED")
        except Exception:
            logging.exception(f"Multi-device: rank dev={did} comparison failure")
            all_precision.append(f"rank{did}:COMPARE_EXCEPTION")


# ---------------------------------------------------------------------------
# Attribute parsing helpers
# ---------------------------------------------------------------------------


def get_transpose_flags(first_ctx):
    attrs = first_ctx.attributes
    tx2 = attrs.get("transposeX2", attrs.get("isTransB", attrs.get("is_trans_b", None)))
    if tx2 is not None:
        return bool(tx2)
    remark = first_ctx.remark or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "is_trans_b":
            try:
                return bool(int(kv[1].strip()))
            except ValueError:
                pass
    return False


# ---------------------------------------------------------------------------
# Quant computation helpers
# ---------------------------------------------------------------------------


def scale_generate(fp32_deq_scale):
    """Apply high-19-bit mask to fp32 scale (simulates hardware)."""
    uint32_deq_scale = numpy.frombuffer(fp32_deq_scale, numpy.uint32)
    uint32_deq_scale &= 0xFFFFE000
    fp32_deq_scale = numpy.frombuffer(uint32_deq_scale, numpy.float32)
    return fp32_deq_scale


def unpack_group_size(group_size):
    """Unpack int64 group_size to (M, N, K) tuple."""
    if group_size == -1 or group_size == 0:
        return 0, 0, 0
    gsm = (group_size >> 32) & 0xFFFF
    gsn = (group_size >> 16) & 0xFFFF
    gsk = group_size & 0xFFFF
    return gsm, gsn, gsk


def per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale):
    """Per-block quantized matmul golden."""
    import torch

    gsm, gsn, gsk = unpack_group_size(group_size)
    if gsm == 0 or gsn == 0 or gsk == 0:
        out = torch.matmul(x1, x2)
        if x1_scale is not None and x2_scale is not None:
            double_scale = scale_generate((x1_scale.numpy() * x2_scale.numpy()))
            out = out * torch.from_numpy(double_scale).float()
        return out
    m = x1.shape[-2]
    k = x1.shape[-1]
    n = x2.shape[-1]
    out = torch.zeros(m, n)
    for m_idx in range((m + gsm - 1) // gsm):
        m_start = m_idx * gsm
        m_end = min((m_idx + 1) * gsm, m)
        for n_idx in range((n + gsn - 1) // gsn):
            n_start = n_idx * gsn
            n_end = min((n_idx + 1) * gsn, n)
            for k_idx in range((k + gsk - 1) // gsk):
                k_start = k_idx * gsk
                k_end = min((k_idx + 1) * gsk, k)
                block_out = (
                    torch.matmul(
                        x1[m_start:m_end, k_start:k_end],
                        x2[k_start:k_end, n_start:n_end],
                    )
                    * x1_scale[m_idx, k_idx]
                    * x2_scale[k_idx, n_idx]
                )
                out[m_start:m_end, n_start:n_end] += block_out
    return out


def mxfp_cpu_compute(x1, x2, x1scale, x2scale):
    """MXFP quantized matmul golden."""
    import numpy as np
    import torch

    if x1scale.ndim == 3:
        x1scale = x1scale.reshape(x1scale.shape[0], -1)
    if x2scale.ndim == 3:
        n_dim = x2.shape[-1]
        if x2scale.shape[1] == n_dim:
            x2scale = np.transpose(x2scale, (0, 2, 1)).reshape(-1, n_dim)
        else:
            x2scale = x2scale.reshape(x2scale.shape[0], -1)
    if x2scale.shape[0] != x1scale.shape[1] and x2scale.shape[1] == x1scale.shape[1]:
        x2scale = x2scale.T
    repeated_x1s = np.repeat(x1scale, 32, axis=-1)
    repeated_x2s = np.repeat(x2scale, 32, axis=-2)
    x1_pad_len = repeated_x1s.shape[-1] - x1.shape[-1]
    x2_pad_len = repeated_x2s.shape[-2] - x2.shape[-2]
    if x1_pad_len > 0:
        x1 = np.pad(x1, [(0, 0)] * (x1.ndim - 1) + [(0, x1_pad_len)], mode="constant")
    if x2_pad_len > 0:
        x2 = np.pad(
            x2, [(0, 0)] * (x2.ndim - 2) + [(0, x2_pad_len), (0, 0)], mode="constant"
        )
    out = np.matmul(x1 * repeated_x1s, x2 * repeated_x2s)
    return out


# ---------------------------------------------------------------------------
# GMM helpers
# ---------------------------------------------------------------------------


def grouped_matmul_cpu(gmm_x, gmm_weight, group_list):
    import torch
    import numpy as np

    B_list = list(torch.unbind(gmm_weight, dim=0))
    A_groups = torch.split(gmm_x, group_list, dim=0)
    results = []
    for i in range(len(group_list)):
        a = A_groups[i].numpy()
        b = B_list[i].numpy()
        results.append(torch.from_numpy(np.matmul(a, b)))
    return torch.cat(results, dim=0)


def quant_grouped_matmul_cpu(
    gmm_x,
    gmm_weight,
    group_list,
    gmm_x_scale=None,
    gmm_weight_scale=None,
    is_mxfp=False,
    is_tt=False,
):
    import torch
    import numpy as np

    if not is_mxfp and not is_tt:
        return grouped_matmul_cpu(gmm_x, gmm_weight, group_list)
    if is_mxfp:
        xs_np = (
            gmm_x_scale if isinstance(gmm_x_scale, np.ndarray) else gmm_x_scale.numpy()
        )
        ws_np = (
            gmm_weight_scale
            if isinstance(gmm_weight_scale, np.ndarray)
            else gmm_weight_scale.numpy()
        )
        xs_np = xs_np.reshape(xs_np.shape[0], -1)
        ep, k_groups, N, pair = ws_np.shape
        ws_np = ws_np.transpose(0, 1, 3, 2).reshape(ep, k_groups * pair, N)
        x_np = gmm_x.numpy()
        w_np = gmm_weight.numpy()
        results = []
        offset = 0
        for i, gl in enumerate(group_list):
            if gl <= 0:
                continue
            x_chunk = x_np[offset : offset + gl]
            xs_chunk = xs_np[offset : offset + gl]
            w_expert = w_np[i]
            ws_expert = ws_np[i]
            rep_x1s = np.repeat(xs_chunk, 32, axis=-1)
            rep_x2s = np.repeat(ws_expert, 32, axis=-2)
            k_pad = rep_x1s.shape[1] - x_chunk.shape[1]
            if k_pad > 0:
                x_chunk = np.pad(x_chunk, ((0, 0), (0, k_pad)))
                w_expert = np.pad(w_expert, ((0, k_pad), (0, 0)))
            out = np.matmul(
                x_chunk * rep_x1s[:, : x_chunk.shape[1]],
                w_expert * rep_x2s[: w_expert.shape[0], :],
            )
            results.append(torch.from_numpy(out))
            offset += gl
        return torch.cat(results, dim=0).to(torch.float32)
    else:
        gmm_out = grouped_matmul_cpu(gmm_x, gmm_weight, group_list)
        combined = gmm_x_scale * gmm_weight_scale
        if combined.dim() == 0:
            combined = combined.unsqueeze(0).unsqueeze(0)
        elif combined.dim() == 1:
            combined = combined.unsqueeze(0)
        gmm_out = gmm_out * combined
        return gmm_out


def get_gmm_exp_token_nums(first_ctx, rank_idx, ep_ws):
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
    return generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)


def generate_gmm_alltoallv_matrix(A_array_val, exp_per_card, seed):
    n = len(A_array_val)
    rng = numpy.random.default_rng(seed)
    total = sum(A_array_val)
    if total % n != 0:
        return [[total // n] * (exp_per_card * n) for _ in range(n)]
    col_sum = total // n
    k_values = []
    for a in A_array_val:
        if a % n != 0:
            return [[col_sum // (exp_per_card)] * (exp_per_card * n) for _ in range(n)]
        k = a // n
        k_values.append(max(k, exp_per_card))
    blocks = []
    for k in k_values:
        block = numpy.zeros((exp_per_card, n), dtype=int)
        for col in range(n):
            counts = rng.multinomial(
                k - exp_per_card, [1.0 / exp_per_card] * exp_per_card
            )
            block[:, col] = counts + 1
        blocks.append(block)
    tmp = numpy.vstack(blocks)
    return [list(col) for col in zip(*tmp)]


def patch_gmm_rank_attributes(ctx, rank_idx, world_size):
    api_name = ctx.api_name
    is_alltoallv_gmm = "AlltoAllvGroupedMatMul" in api_name
    is_gmm_alltoallv = "GroupedMatMulAlltoAllv" in api_name
    if not is_alltoallv_gmm and not is_gmm_alltoallv:
        return
    attrs = ctx.attributes
    ep_ws = attrs.get("epWorldSize", world_size)
    exp_per_card = (
        ctx.tensor_view_shapes[1][0] if len(ctx.tensor_view_shapes) > 1 else 1
    )
    seed_val = 0
    remark = ctx.remark or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "seed":
            try:
                seed_val = int(kv[1].strip())
            except ValueError:
                pass
    if is_alltoallv_gmm:
        bsk = ctx.tensor_view_shapes[0][0] if ctx.tensor_view_shapes else 0
        A_array = [bsk] * ep_ws
        expTokenNums = generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)
        send_counts = expTokenNums[rank_idx]
        recv_counts = []
        for i in range(ep_ws):
            recv_counts.extend(
                expTokenNums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card]
            )
        attrs["sendCounts"] = send_counts
        attrs["recvCounts"] = recv_counts
    elif is_gmm_alltoallv:
        M_per_rank = ctx.tensor_view_shapes[0][0] if ctx.tensor_view_shapes else 0
        A_array = [M_per_rank] * ep_ws
        expTokenNums = generate_gmm_alltoallv_matrix(A_array, exp_per_card, seed_val)
        recv_counts = expTokenNums[rank_idx]
        send_counts = []
        for i in range(ep_ws):
            send_counts.extend(
                expTokenNums[i][rank_idx * exp_per_card : (rank_idx + 1) * exp_per_card]
            )
        attrs["sendCounts"] = send_counts
        attrs["recvCounts"] = recv_counts
        ctx._pure_attrs = None
        logging.info(
            f"[GMM patch] api={api_name} rank={rank_idx} ep_ws={ep_ws} "
            f"seed={seed_val} send_counts={send_counts[:4]}... recv_counts={recv_counts[:4]}..."
        )


# ---------------------------------------------------------------------------
# GMM alltoallv golden compute (moved from TTK __golden_gmm_alltoallv)
# ---------------------------------------------------------------------------


def _get_gmm_send_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws):
    group_list = []
    for j in range(exp_per_card):
        total = sum(expTokenNums[rank_idx][r * exp_per_card + j] for r in range(ep_ws))
        group_list.append(total)
    return group_list


def _get_gmm_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws):
    group_list = []
    for j in range(exp_per_card):
        total = sum(expTokenNums[i][rank_idx * exp_per_card + j] for i in range(ep_ws))
        group_list.append(total)
    return group_list


def _unpermute_gmm_alltoallv(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    import numpy as np

    send_gl = _get_gmm_send_group_list(expTokenNums, rank_idx, exp_per_card, ep_ws)
    expert_offsets = np.concatenate([[0], np.cumsum(send_gl[:-1])])
    my_row = expTokenNums[rank_idx]
    per_expert_cumsum = np.zeros((exp_per_card, ep_ws), dtype=np.int64)
    for j in range(exp_per_card):
        for r in range(ep_ws):
            per_expert_cumsum[j][r] = my_row[r * exp_per_card + j]
    per_expert_cumsum = np.cumsum(per_expert_cumsum, axis=1)
    all_indices = []
    for r in range(ep_ws):
        for j in range(exp_per_card):
            start = int(per_expert_cumsum[j][r - 1]) if r > 0 else 0
            end = int(per_expert_cumsum[j][r])
            all_indices.extend(
                range(int(expert_offsets[j]) + start, int(expert_offsets[j]) + end)
            )
    if len(all_indices) == 0:
        return tokens.clone()
    idx_tensor = torch.tensor(all_indices, dtype=torch.long)
    return tokens.index_select(0, idx_tensor).to(tokens.dtype)


def _permute_alltoallv_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    import numpy as np

    indices = np.zeros((exp_per_card, ep_ws), dtype=np.int64)
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = int(expTokenNums[i][j + (exp_per_card * rank_idx)])
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
        return tokens.clone()
    idx_tensor = torch.tensor(all_indices, dtype=torch.long)
    return tokens.index_select(0, idx_tensor)


def _permute_a2a_gmm(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch

    indices = torch.zeros(exp_per_card, ep_ws).long()
    for j in range(exp_per_card):
        for i in range(ep_ws):
            indices[j][i] = expTokenNums[i][j + exp_per_card * rank_idx]
    trans = indices.permute(1, 0)
    flaten = trans.reshape(-1)
    sum_list = torch.cumsum(flaten, dim=0)
    tmp = []
    for i in range(len(sum_list)):
        if i == 0:
            tmp.append(range(0, sum_list[i]))
        else:
            tmp.append(range(sum_list[i - 1], sum_list[i]))
    parts = []
    expert_sizes = []
    for e in range(exp_per_card):
        exp_token = []
        for r in range(ep_ws):
            exp_token += list(tmp[e + r * exp_per_card])
        combined = torch.tensor(exp_token)
        parts.append(tokens.index_select(0, combined))
        expert_sizes.append(len(exp_token))
    K = tokens.shape[1] if tokens.dim() > 1 else 1
    result = torch.zeros(sum(expert_sizes), K, dtype=tokens.dtype)
    offset = 0
    for e in range(exp_per_card):
        result[offset : offset + expert_sizes[e]] = parts[e]
        offset += expert_sizes[e]
    return result, expert_sizes


def _unpermute_mc2(tokens, exp_per_card, ep_ws, rank_idx, expTokenNums):
    import torch
    import numpy as np

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


def _simulate_alltoallv(
    all_inputs,
    device_ids,
    send_counts_per_rank,
    recv_counts_per_rank,
    ep_ws,
    exp_per_card,
):
    import torch

    rank_outputs = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        received_chunks = []
        offset = 0
        for src_did in device_ids:
            src_idx = list(device_ids).index(src_did)
            src_data = all_inputs[src_did]
            src_send = send_counts_per_rank[src_did]
            chunk_start = offset
            chunk_size = sum(
                src_send[target_idx * exp_per_card : (target_idx + 1) * exp_per_card]
            )
            chunk = src_data[chunk_start : chunk_start + chunk_size]
            received_chunks.append(chunk)
            offset += chunk_size
        rank_outputs[target_did] = (
            torch.cat(received_chunks, dim=0)
            if received_chunks
            else torch.zeros(0, src_data.shape[1] if src_data.dim() > 1 else 0)
        )
    return rank_outputs


def golden_gmm_alltoallv(
    thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card
):
    """Compute golden for GroupedMatMulAlltoAllv."""
    import torch
    import numpy as np

    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    trans_gmm_weight = attrs.get("transGmmWeight", False)
    trans_mm_weight = attrs.get("transMmWeight", False)

    gmm_x_qm = int(attrs.get("gmmXQuantMode", 0))
    gmm_w_qm = int(attrs.get("gmmWeightQuantMode", 0))
    is_mxfp = gmm_x_qm == 6 and gmm_w_qm == 6
    is_tt = gmm_x_qm == 1 or gmm_w_qm == 1
    is_quant = is_mxfp or is_tt

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
        if is_quant:
            gmm_x_scale = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            gmm_w_scale = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            if is_mxfp:
                xs_np = e8m0_to_f32(gmm_x_scale)
                ws_np = e8m0_to_f32(gmm_w_scale)
                if isinstance(xs_np, np.ndarray):
                    xs_np = torch.from_numpy(xs_np)
                if isinstance(ws_np, np.ndarray):
                    ws_np = torch.from_numpy(ws_np)
                xs_np = xs_np.numpy()
                ws_np = ws_np.numpy()
                gmm_out = quant_grouped_matmul_cpu(
                    gmm_x, gmm_weight, recv_gl, xs_np, ws_np, is_mxfp=True, is_tt=False
                )
            else:
                xs_f = to_torch_f32(gmm_x_scale)
                ws_f = to_torch_f32(gmm_w_scale)
                gmm_out = quant_grouped_matmul_cpu(
                    gmm_x, gmm_weight, recv_gl, xs_f, ws_f, is_mxfp=False, is_tt=True
                )
        else:
            gmm_out = grouped_matmul_cpu(gmm_x, gmm_weight, recv_gl)
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

        output_splits = []
        for i in range(ep_ws):
            output_splits.append(
                sum(
                    expTokenNums[i][
                        target_rank * exp_per_card : (target_rank + 1) * exp_per_card
                    ]
                )
            )

        input_splits_map = {}
        for src_did in device_ids:
            src_rank = list(device_ids).index(src_did)
            is_list = []
            for t in range(ep_ws):
                is_list.append(
                    sum(
                        expTokenNums[src_rank][
                            t * exp_per_card : (t + 1) * exp_per_card
                        ]
                    )
                )
            input_splits_map[src_did] = is_list

        output_chunks = []
        for src_did in device_ids:
            src_rank = list(device_ids).index(src_did)
            src_unpermuted = all_unpermuted[src_did]
            is_list = input_splits_map[src_did]
            offset = 0
            for t in range(ep_ws):
                if t == target_rank:
                    chunk = src_unpermuted[offset : offset + is_list[t]]
                    output_chunks.append(chunk.clone())
                offset += is_list[t]

        main_golden = (
            torch.cat(output_chunks, dim=0) if output_chunks else torch.zeros(0, N)
        )
        rank_goldens[target_did] = {"main": main_golden}
        del output_chunks, main_golden

        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_x_f = to_torch_f32(mm_x)
            mm_weight_f = to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            rank_goldens[target_did]["mm"] = torch.mm(mm_x_f, mm_weight_f)
            del mm_x_f, mm_weight_f
        else:
            rank_goldens[target_did]["mm"] = None

    del all_gmm_out, all_unpermuted
    return rank_goldens


def golden_alltoallv_gmm(
    thread_contexts, device_ids, expTokenNums, ep_ws, exp_per_card
):
    """Compute golden for AlltoAllvGroupedMatMul."""
    import torch
    import numpy as np

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
        recv_by_src = output_splits
        recv_cumsum = list(np.cumsum(recv_by_src))
        recv_offsets = [0] + recv_cumsum[:-1]
        K = (
            all_a2a_inputs[device_ids[0]].shape[1]
            if all_a2a_inputs[device_ids[0]].dim() > 1
            else 1
        )
        gathered = torch.zeros(sum(recv_by_src), K)
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

        gmm_out = grouped_matmul_cpu(permuted, gmm_weight, expert_sizes)
        del gmm_weight

        rank_goldens[did] = {}
        rank_goldens[did]["main"] = gmm_out.contiguous()
        rank_goldens[did]["permute"] = (
            permuted.contiguous() if permute_out_flag else None
        )
        del gmm_out, permuted

        mm_x = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        mm_weight = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
        if mm_x is not None and isinstance(mm_x, torch.Tensor) and mm_x.numel() > 0:
            mm_x_f = to_torch_f32(mm_x)
            mm_weight_f = to_torch_f32(mm_weight)
            if trans_mm_weight:
                mm_weight_f = mm_weight_f.t().contiguous()
            mm_golden = torch.mm(mm_x_f, mm_weight_f)
            rank_goldens[did]["mm"] = mm_golden
            del mm_x_f, mm_weight_f, mm_golden
        else:
            rank_goldens[did]["mm"] = None

    del all_a2a_inputs, all_send_segments, a2a_outputs
    return rank_goldens


# ---------------------------------------------------------------------------
# AllGather matmul golden (V1 + V2 quant variants)
# ---------------------------------------------------------------------------


def golden_all_gather_compare(thread_contexts, device_ids, all_precision, world_size):
    """AllGatherMatmul golden: all_gather(x1) -> matmul(gathered, x2) -> output.

    Handles V1 (non-quant) and V2 (quant: per_tensor / mxfp / per_block).
    """
    import torch
    import numpy as np

    first_ctx = next(iter(thread_contexts.values()))
    api_name = first_ctx.api_name or ""
    is_v2 = "V2" in api_name or "v2" in api_name

    remark = first_ctx.remark or ""
    per_block_flag = False
    is_mxfp = False
    is_bias = False
    gather_output = len(first_ctx.output_tensor_indexes or ()) > 1
    is_trans_b = False
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2:
            k, v = kv[0].strip(), kv[1].strip()
            if k == "per_block_flag":
                per_block_flag = v.lower() in ("1", "true")
            elif k == "is_trans_b":
                is_trans_b = v == "1"
            elif k == "is_mxfp":
                is_mxfp = v.lower() in ("1", "true")
            elif k == "is_bias":
                is_bias = v == "1"
            elif k == "gather_output":
                gather_output = v == "1"

    x1_dtype_str = (
        first_ctx.flat_tensor_dtypes[0] if first_ctx.flat_tensor_dtypes else ""
    )
    is_quant = x1_dtype_str in (
        "fp8_e4m3fn",
        "fp8_e5m2",
        "hif8",
        "float8_e4m3fn",
        "float8_e5m2",
        "hifloat8",
    )

    if is_v2 and is_quant and len(first_ctx.flat_tensor_dtypes) > 3:
        x1s_dtype = first_ctx.flat_tensor_dtypes[3]
        if x1s_dtype in ("fp8_e8m0", "float8_e8m0"):
            is_mxfp = True

    def _to_torch_keep_dtype_npu_scale(t):
        if t is None:
            return None
        if isinstance(t, np.ndarray):
            dtype_name = str(t.dtype)
            if "e8m0" in dtype_name:
                return torch.from_numpy(t.view(np.uint8))
            return torch.from_numpy(t.astype(np.float32, copy=False))
        dtype_name = str(t.dtype)
        if "e8m0" in dtype_name or (t.dtype == torch.uint8):
            return t
        return t.float()

    all_x1 = []
    x2_per_rank = {}
    bias_per_rank = {}
    x1scale_per_rank = {}
    x2scale_per_rank = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        all_x1.append(to_torch_f32(x1))
        x2_per_rank[did] = to_torch_f32(x2)
        if len(tc.flatten_tensors) > 2 and tc.flatten_tensors[2] is not None:
            bias_per_rank[did] = to_torch_f32(tc.flatten_tensors[2])
        else:
            bias_per_rank[did] = None
        if is_v2 and is_quant:
            if len(tc.flatten_tensors) > 3 and tc.flatten_tensors[3] is not None:
                if is_mxfp or per_block_flag:
                    x1scale_per_rank[did] = _to_torch_keep_dtype_npu_scale(
                        tc.flatten_tensors[3]
                    )
                else:
                    x1scale_per_rank[did] = to_torch_f32(tc.flatten_tensors[3])
            if len(tc.flatten_tensors) > 4 and tc.flatten_tensors[4] is not None:
                if is_mxfp or per_block_flag:
                    x2scale_per_rank[did] = _to_torch_keep_dtype_npu_scale(
                        tc.flatten_tensors[4]
                    )
                else:
                    x2scale_per_rank[did] = to_torch_f32(tc.flatten_tensors[4])
    gathered = torch.cat(all_x1, dim=0)
    del all_x1

    if is_v2 and is_quant and (per_block_flag or is_mxfp) and x1scale_per_rank:
        all_x1s = [
            x1scale_per_rank[did] for did in device_ids if did in x1scale_per_rank
        ]
        if all_x1s:
            gathered_x1scale = torch.cat(all_x1s, dim=0)
        else:
            gathered_x1scale = None
    else:
        gathered_x1scale = None

    rank_goldens = {}
    for did in device_ids:
        x2_f = x2_per_rank[did]
        bias = bias_per_rank.get(did)

        if not is_quant:
            golden = torch.matmul(gathered, x2_f)
            if bias is not None:
                golden = golden + bias
        else:
            x1s = x1scale_per_rank.get(did)
            x2s = x2scale_per_rank.get(did)
            if per_block_flag:
                gs = gathered_x1scale
                group_size = first_ctx.attributes.get("groupSize", 0)
                golden = per_block_cpu_compute(group_size, gathered, x2_f, gs, x2s)
            elif is_mxfp:
                gs = gathered_x1scale
                x2_for_golden = x2_f
                golden = mxfp_cpu_compute(
                    gathered.numpy().astype(np.float32),
                    x2_for_golden.numpy().astype(np.float32),
                    gs.numpy().astype(np.float32),
                    x2s.numpy().astype(np.float32),
                )
                golden = torch.from_numpy(golden)
                if bias is not None:
                    golden = golden + bias
            else:
                golden = torch.matmul(gathered, x2_f)
                if bias is not None:
                    golden = golden + bias
                double_scale = scale_generate((x1s.numpy() * x2s.numpy()))
                double_scale_t = torch.unsqueeze(
                    torch.from_numpy(double_scale), dim=1
                ).float()
                golden = golden * double_scale_t
        rank_goldens[did] = {
            "main": golden.contiguous(),
            "gather": gathered.contiguous(),
        }
        del golden

    # Cascade third_party
    rank_third_parties = None
    try:
        if not is_quant:
            from ttk.core_modules.npu.op_api.hccl_cascade import (
                run_allgather_matmul_cascade,
            )

            cascade_outs = run_allgather_matmul_cascade(
                thread_contexts,
                device_ids,
                is_trans_b=is_trans_b,
                is_gather_output=gather_output,
            )
        else:
            from ttk.core_modules.npu.op_api.hccl_cascade import (
                run_allgather_quant_matmul_v2_cascade,
            )

            cascade_outs = run_allgather_quant_matmul_v2_cascade(
                thread_contexts,
                device_ids,
                is_trans_b=is_trans_b,
                is_bias=is_bias,
                is_mxfp=is_mxfp,
                per_block_flag=per_block_flag,
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

    del gathered, x2_per_rank
    apply_goldens_and_compare(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


# ---------------------------------------------------------------------------
# ReduceScatter matmul golden
# ---------------------------------------------------------------------------


def golden_reduce_scatter_compare(
    thread_contexts, device_ids, all_precision, world_size
):
    """MatmulReduceScatter golden: matmul(x1, x2) -> reduce_scatter(SUM) -> output."""
    import torch

    first_ctx = next(iter(thread_contexts.values()))

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() == 0:
            bias = None
        x1_f = to_torch_f32(x1)
        x2_f = to_torch_f32(x2)
        mm_out = torch.matmul(x1_f, x2_f)
        if bias is not None:
            mm_out = mm_out + to_torch_f32(bias)
        local_results[did] = mm_out
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

    rank_third_parties = None
    try:
        from ttk.core_modules.npu.op_api.hccl_cascade import (
            run_matmul_reducescatter_cascade,
        )

        remark = first_ctx.remark or ""
        is_trans_b = False
        for part in remark.split(","):
            kv = part.split("=", 1)
            if len(kv) == 2 and kv[0].strip() == "is_trans_b":
                is_trans_b = kv[1].strip() == "1"
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

    apply_goldens_and_compare(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


# ---------------------------------------------------------------------------
# AllReduce matmul golden (V1/V2/V3, WeightQuant, QuantMatmul)
# ---------------------------------------------------------------------------


def golden_all_reduce_compare(thread_contexts, device_ids, all_precision, world_size):
    """AllReduce-type ops: each rank computes locally, then all_reduce(sum)."""
    import torch

    first_ctx = next(iter(thread_contexts.values()))
    api_name = first_ctx.api_name
    attrs = first_ctx.attributes

    is_weight_quant = "WeightQuantMatmulAllReduce" in api_name
    is_quant_matmul = "QuantMatmulAllReduce" in api_name and "Weight" not in api_name
    is_v2 = "AllReduceV2" in api_name or "AllReduceV3" in api_name

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = tc.flatten_tensors[0]
        x2 = tc.flatten_tensors[1]

        if is_weight_quant:
            x1_f = to_torch_f32(x1)
            x2_f = to_torch_f32(x2)
            antiquant_scale = (
                tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
            )
            antiquant_offset = (
                tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
            )
            aq_scale_f = (
                to_torch_f32(antiquant_scale) if antiquant_scale is not None else None
            )
            aq_offset_f = (
                to_torch_f32(antiquant_offset)
                if antiquant_offset is not None
                and isinstance(antiquant_offset, torch.Tensor)
                and antiquant_offset.numel() > 0
                else None
            )
            group_size = int(attrs.get("antiquantGroupSize", 0))
            if group_size > 0 and aq_scale_f is not None:
                aq_scale_f = aq_scale_f.repeat_interleave(group_size, dim=0)
                if aq_offset_f is not None:
                    aq_offset_f = aq_offset_f.repeat_interleave(group_size, dim=0)
            if aq_offset_f is not None:
                weight_deq = (x2_f + aq_offset_f) * aq_scale_f
            elif aq_scale_f is not None:
                weight_deq = x2_f * aq_scale_f
            else:
                weight_deq = x2_f
            mm_out = torch.matmul(x1_f, weight_deq)
        elif is_quant_matmul:
            is_v4_v5 = (
                "QuantMatmulAllReduceV4" in api_name
                or "QuantMatmulAllReduceV5" in api_name
            )
            if is_v4_v5:
                x1_f = to_torch_f32(x1)
                x2_f = to_torch_f32(x2)
                mm_out = torch.matmul(x1_f, x2_f)
                x1scale = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                x2scale = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
                if x1scale is not None:
                    x1s_f = to_torch_f32(x1scale)
                    if x1s_f.dim() == 1 and mm_out.dim() == 2:
                        x1s_f = x1s_f.unsqueeze(-1)
                    elif x1s_f.dim() == 1 and mm_out.dim() == 3:
                        x1s_f = x1s_f.unsqueeze(0).unsqueeze(-1)
                    mm_out = mm_out * x1s_f
                if x2scale is not None:
                    x2s_f = to_torch_f32(x2scale)
                    mm_out = mm_out * x2s_f
                ds = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
                if ds is not None:
                    ds_f = to_torch_f32(ds)
                    if ds_f.dim() == 1 and mm_out.dim() >= 2:
                        ds_f = ds_f.unsqueeze(0)
                    mm_out = mm_out * ds_f
                x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                if x3 is not None and hasattr(x3, "numel") and x3.numel() > 0:
                    mm_out = mm_out + to_torch_f32(x3)
            else:
                x1_f = to_torch_f32(x1)
                x2_f = to_torch_f32(x2)
                mm_out = torch.matmul(x1_f, x2_f)
                is_v2_quant = "QuantMatmulAllReduceV2" in api_name
                if is_v2_quant:
                    ds = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                    if ds is not None:
                        ds_f = to_torch_f32(ds)
                        mm_out = mm_out * ds_f
                    pt = tc.flatten_tensors[5] if len(tc.flatten_tensors) > 5 else None
                    if (
                        pt is not None
                        and isinstance(pt, torch.Tensor)
                        and pt.numel() > 0
                    ):
                        pt_f = to_torch_f32(pt)
                        if pt_f.dim() == 1:
                            pt_f = pt_f.unsqueeze(1)
                        mm_out = mm_out * pt_f
                    x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                    if (
                        x3 is not None
                        and isinstance(x3, torch.Tensor)
                        and x3.numel() > 0
                    ):
                        mm_out = mm_out + to_torch_f32(x3)
                else:
                    ds = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
                    if ds is not None:
                        ds_f = to_torch_f32(ds)
                        mm_out = mm_out * ds_f
                    bias = (
                        tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
                    )
                    if (
                        bias is not None
                        and isinstance(bias, torch.Tensor)
                        and bias.numel() == 0
                    ):
                        bias = None
                    if bias is not None:
                        mm_out = mm_out + to_torch_f32(bias)
                    if "V3" in api_name and len(tc.flatten_tensors) > 5:
                        pt = tc.flatten_tensors[5]
                        if (
                            pt is not None
                            and isinstance(pt, torch.Tensor)
                            and pt.numel() > 0
                        ):
                            pt_f = to_torch_f32(pt)
                            if pt_f.dim() == 1:
                                pt_f = pt_f.unsqueeze(1)
                            mm_out = mm_out * pt_f
        else:
            bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            if (
                bias is not None
                and isinstance(bias, torch.Tensor)
                and bias.numel() == 0
            ):
                bias = None
            x1_f = to_torch_f32(x1)
            x2_f = to_torch_f32(x2)
            mm_out = torch.matmul(x1_f, x2_f)
            if bias is not None:
                mm_out = mm_out + to_torch_f32(bias)
            if is_v2:
                x3 = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
                if x3 is not None and isinstance(x3, torch.Tensor) and x3.numel() > 0:
                    mm_out = mm_out + to_torch_f32(x3)

        x1_dtype = x1.dtype if hasattr(x1, "dtype") else None
        if x1_dtype is not None and x1_dtype in (torch.bfloat16, torch.float16):
            mm_out = mm_out.to(x1_dtype).float()
        local_results[did] = mm_out

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    rank_third_parties = None
    try:
        from ttk.core_modules.npu.op_api.hccl_cascade import (
            run_matmul_allreduce_cascade,
        )

        t_x1 = False
        t_x2 = get_transpose_flags(first_ctx)
        if is_weight_quant:
            t_x2 = False
        is_bias_flag = False
        for did in device_ids:
            tc = thread_contexts[did]
            if is_weight_quant:
                break
            if is_quant_matmul:
                b = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
            else:
                b = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
            if b is not None and isinstance(b, torch.Tensor) and b.numel() > 0:
                is_bias_flag = True
                break
        cascade_outs = run_matmul_allreduce_cascade(
            thread_contexts,
            device_ids,
            transpose_x1=t_x1,
            transpose_x2=t_x2,
            is_bias=is_bias_flag,
        )
        rank_third_parties = {did: [cascade_outs[did]["main"]] for did in device_ids}
        logging.info("MatmulAllReduce: real HCCL cascade succeeded")
    except Exception:
        logging.exception("MatmulAllReduce: real HCCL cascade failed, no third_party")
        rank_third_parties = None

    rank_goldens = {did: total for did in device_ids}
    apply_goldens_and_compare(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


# ---------------------------------------------------------------------------
# MatmulAlltoAll / AlltoAllMatmul golden compute
# ---------------------------------------------------------------------------


def golden_matmul_allto_all(
    thread_contexts,
    device_ids,
    target_did,
    x1,
    x2,
    bias,
    t_x1,
    t_x2,
    world_size,
    x1scale=None,
    x2scale=None,
):
    """MatmulAlltoAll: matmul(x1, x2) -> [dequant scales] -> all_to_all -> output."""
    import torch

    input_mat = to_torch_f32(x1)
    if t_x1:
        input_mat = input_mat.t().contiguous()
    weight_mat = to_torch_f32(x2)
    if t_x2:
        weight_mat = weight_mat.t().contiguous()
    mm_out = torch.matmul(input_mat, weight_mat)
    if bias is not None:
        mm_out = mm_out + to_torch_f32(bias)
    if x1scale is not None:
        x1s_f = to_torch_f32(x1scale)
        if x1s_f.dim() == 1:
            x1s_f = x1s_f.unsqueeze(-1)
        mm_out = mm_out * x1s_f
    if x2scale is not None:
        x2s_f = to_torch_f32(x2scale)
        if x2s_f.dim() == 1:
            x2s_f = x2s_f.unsqueeze(0)
        mm_out = mm_out * x2s_f

    M = mm_out.shape[0]
    N = mm_out.shape[1]
    chunk_n = N // world_size

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
        s_chunks = s_mm.view(M, world_size, chunk_n).permute(1, 0, 2).contiguous()
        s_chunks = s_chunks.view(world_size, M * chunk_n)
        send_chunks = s_chunks.chunk(world_size, dim=0)
        target_idx = list(device_ids).index(target_did)
        all_to_all_results.append(send_chunks[target_idx].clone())
        del s_mm, s_chunks, send_chunks

    received = torch.cat(all_to_all_results, dim=0)
    del all_to_all_results
    received = received.reshape(-1, chunk_n).contiguous()
    return {"main": received}


def golden_allto_all_matmul(
    thread_contexts, device_ids, target_did, x1, x2, bias, t_x1, t_x2, world_size
):
    """AlltoAllMatmul: all_to_all(x1) -> matmul(a2a_out, x2) -> output."""
    import torch

    input_mat = to_torch_f32(x1)
    if t_x1:
        input_mat = input_mat.t().contiguous()
    weight_mat = to_torch_f32(x2)
    if t_x2:
        weight_mat = weight_mat.t().contiguous()
    M_total = input_mat.shape[0]
    K = input_mat.shape[1]
    M_chunk = M_total // world_size

    target_idx = list(device_ids).index(target_did)

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
    a2a_out = recv_tensor.permute(1, 0, 2).reshape(M_chunk, world_size * K).contiguous()

    mm_out = torch.matmul(a2a_out, weight_mat)
    if bias is not None:
        mm_out = mm_out + to_torch_f32(bias)
    return {"main": mm_out, "alltoall": a2a_out}


# ---------------------------------------------------------------------------
# MoE alltoallv simulation
# ---------------------------------------------------------------------------


def simulate_moe_alltoallv(all_rank_inputs, device_ids, send_counts_per_rank):
    import torch

    rank_outputs = {}
    for target_did in device_ids:
        target_idx = list(device_ids).index(target_did)
        received_chunks = []
        for src_did in device_ids:
            src_data = all_rank_inputs[src_did]
            src_counts = send_counts_per_rank[src_did]
            offset = 0
            for dst_idx in range(len(device_ids)):
                if dst_idx == target_idx:
                    count = int(src_counts[dst_idx])
                    if count > 0:
                        received_chunks.append(src_data[offset : offset + count])
                    break
                offset += int(src_counts[dst_idx])
        if received_chunks:
            rank_outputs[target_did] = torch.cat(received_chunks, dim=0)
        else:
            h = (
                all_rank_inputs[device_ids[0]].shape[-1]
                if all_rank_inputs[device_ids[0]].dim() > 1
                else 0
            )
            rank_outputs[target_did] = torch.zeros(
                0, h, dtype=all_rank_inputs[device_ids[0]].dtype
            )
    return rank_outputs


def generate_exp_token_nums(exp_num, ep_world_size, bsk, seed):
    m = exp_num
    n = ep_world_size
    total = bsk
    sum_row = total * n // m
    sum_col = total
    if m * sum_row != n * sum_col:
        return [[total // m] * m for _ in range(n)]
    numpy.random.seed(seed)
    matrix = numpy.random.multinomial(sum_row - n, [1.0 / n] * n, size=m) + 1
    cur_col = matrix.sum(axis=0)
    target = numpy.full(n, sum_col)
    for _ in range(10000):
        if numpy.array_equal(cur_col, target):
            break
        j = int(numpy.argmax(cur_col - target))
        k = int(numpy.argmin(cur_col - target))
        for i in numpy.random.permutation(m):
            if matrix[i, j] > 1:
                matrix[i, j] -= 1
                matrix[i, k] += 1
                cur_col[j] -= 1
                cur_col[k] += 1
                break
    return [list(col) for col in zip(*matrix)]
