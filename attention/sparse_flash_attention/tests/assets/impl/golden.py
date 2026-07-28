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

import math
import importlib.util
import sys
from pathlib import Path

import torch


PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    """Load tests/pytest/sparse_flash_attention_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "sparse_flash_attention_golden.py"
    saved_generate_tensor_data = sys.modules.pop("generate_tensor_data", None)
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"sfa_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
        sys.modules.pop("generate_tensor_data", None)
        if saved_generate_tensor_data is not None:
            sys.modules["generate_tensor_data"] = saved_generate_tensor_data
    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


def compute_golden(input_tensor_dict, params):
    return load_pytest_golden_module().compute_golden(input_tensor_dict, params)


# -----------------------------------------------------------------------------
# TTK TestSpec adapter appended below. The pytest CPU formula is loaded lazily;
# the wrapper below adapts TTK framework-API inputs and attrs to it.
# -----------------------------------------------------------------------------

__golden__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "cpu_sparse_flash_attention",
    }
}


def as_cpu(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu() if torch.is_tensor(tensor) else tensor


def to_cpu_float32_tensor(tensor):
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return tensor
    cpu = tensor.detach().cpu()
    if cpu.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return cpu.float()
    return cpu.float() if cpu.is_floating_point() else cpu


def to_int_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    return [int(v) for v in value]


def normalize_tnd_cumulative_seq(seq_list, total_t):
    if not seq_list:
        return [int(total_t)]
    seq_list = [int(v) for v in seq_list]
    if (
        all(seq_list[i] >= seq_list[i - 1] for i in range(1, len(seq_list)))
        and seq_list[-1] == total_t
    ):
        return seq_list

    fixed = []
    prev = 0
    for value in seq_list:
        value = min(max(value, prev), int(total_t))
        fixed.append(value)
        prev = value
    fixed[-1] = int(total_t)
    return fixed


def dtype_name(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return None
    mapping = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    return mapping.get(tensor.dtype, str(tensor.dtype))


def shape_of(tensor, default=None):
    if tensor is None:
        return default
    return tuple(tensor.shape)


def default_rope_like(ref_tensor, rope_dim=64):
    if ref_tensor is None:
        return None
    shape = list(ref_tensor.shape)
    shape[-1] = int(rope_dim)
    return torch.zeros(shape, dtype=ref_tensor.dtype, device=ref_tensor.device)


def pa_to_bnsd(key_pa, block_table, actual_seq_kv, batch, block_size):
    """Restore TTK PA cache `(block_num, block_size, N, D)` to BSND."""
    key_pa = as_cpu(key_pa)
    block_table = as_cpu(block_table)
    actual_seq_kv = to_int_list(actual_seq_kv)
    batch = int(batch)
    block_size = int(block_size)

    num_heads = key_pa.shape[2]
    dim = key_pa.shape[3]
    max_seq = (
        max(actual_seq_kv)
        if actual_seq_kv
        else block_size * (block_table.shape[1] if block_table.dim() > 1 else 1)
    )
    out = torch.zeros((batch, max_seq, num_heads, dim), dtype=key_pa.dtype)

    for b_idx in range(batch):
        cur_len = actual_seq_kv[b_idx] if b_idx < len(actual_seq_kv) else max_seq
        table_width = block_table.shape[1] if block_table.dim() > 1 else 1
        for page_idx in range(table_width):
            block_id = (
                int(block_table[b_idx, page_idx].item())
                if block_table.dim() > 1
                else int(block_table[b_idx].item())
            )
            if block_id < 0 or block_id >= key_pa.shape[0]:
                continue
            offset = page_idx * block_size
            if offset >= cur_len:
                break
            take = min(block_size, cur_len - offset, key_pa.shape[1])
            out[b_idx, offset : offset + take, :, :] = key_pa[block_id, :take, :, :]
    return out


def ttk_gather_kv(
    k_tensor,
    v_tensor,
    sparse_indices,
    sparse_blocksize,
    sparse_blockcount,
    batch,
    n2_idx,
    s1_idx,
    curr_actual_seq,
    curr_actual_seq_q,
    sparse_mode,
):
    # Same semantics as pytest gatherKV, plus bounds checks for provided TTK sparse_indices.
    if sparse_mode == 0:
        threshold = int(curr_actual_seq)
    elif sparse_mode == 3:
        threshold = int(curr_actual_seq) - int(curr_actual_seq_q) + int(s1_idx) + 1
    else:
        threshold = int(curr_actual_seq)

    threshold = max(0, min(threshold, int(k_tensor.shape[2])))
    valid_count = (
        min(int(sparse_blockcount), math.ceil(threshold / int(sparse_blocksize)))
        if threshold > 0
        else 0
    )
    sparse_positions = []
    for i in range(valid_count):
        sparse_id = int(
            sparse_indices[i].item()
            if torch.is_tensor(sparse_indices[i])
            else sparse_indices[i]
        )
        if sparse_id == -1:
            break
        max_block_id = (threshold - 1) // int(sparse_blocksize)
        sparse_id = min(max(sparse_id, 0), max_block_id)
        begin = sparse_id * int(sparse_blocksize)
        end = min(begin + int(sparse_blocksize), threshold)
        if begin < end:
            sparse_positions.extend(range(begin, end))

    if not sparse_positions:
        return True, [], []
    return (
        False,
        k_tensor[batch, n2_idx, sparse_positions, :],
        v_tensor[batch, n2_idx, sparse_positions, :],
    )


# Patch the pytest gatherKV entry at runtime. The pytest formula resolves it
# dynamically, so assets only bounds provided TTK sparse indices.
gatherKV = ttk_gather_kv


def ttk_trans_tnd_actseq(seq):
    if not seq:
        return []
    out = [int(seq[0])]
    for idx in range(len(seq) - 1):
        out.append(max(0, int(seq[idx + 1]) - int(seq[idx])))
    return out


trans_tnd_actseq = ttk_trans_tnd_actseq


def ttk_increattention_bnsd(fa_param):
    batch_size = fa_param["b"]
    num_heads = fa_param["numHeads"]
    num_kv_heads = fa_param["numKeyValueHeads"]
    actual_q = fa_param["actualSeqLengths_q"]
    actual_kv = fa_param["actualSeqLengths_kv"]
    scale_value = fa_param["scaleValue"]
    sparse_blocksize = fa_param["sparse_blocksize"]
    sparse_blockcount = fa_param["sparse_blockcount"]
    sparse_indices = fa_param["sparse_indices_bnsd_tensor"]
    out_shape_bnsd = list(fa_param["q_bnsd_shape"])
    out_shape_bnsd[-1] = fa_param["v_bnsd_shape"][-1]
    sparse_mode = fa_param["sparse_mode"]
    group = num_heads // num_kv_heads

    q_bnsd = fa_param["q_bnsd_tensor"].float()
    k_bnsd = fa_param["k_bnsd_tensor"].float()
    v_bnsd = fa_param["v_bnsd_tensor"].float()

    y = torch.zeros(out_shape_bnsd, dtype=torch.float32)
    softmax_max = torch.zeros(fa_param["softmax_max_shape"], dtype=torch.float32)
    softmax_sum = torch.zeros(fa_param["softmax_sum_shape"], dtype=torch.float32)

    for batch in range(batch_size):
        cur_q = actual_q[batch]
        cur_kv = actual_kv[batch]
        q_offset = 0 if batch == 0 else sum(actual_q[0:batch])
        for kv_head in range(num_kv_heads):
            for q_idx in range(cur_q):
                if q_idx < cur_q - cur_kv and sparse_mode != 0:
                    continue
                q_cur = q_bnsd[batch, kv_head * group : (kv_head + 1) * group, q_idx, :]
                sparse_cur = sparse_indices[batch, kv_head, q_idx, :]
                empty, k_sparse, v_sparse = ttk_gather_kv(
                    k_bnsd,
                    v_bnsd,
                    sparse_cur,
                    sparse_blocksize,
                    sparse_blockcount,
                    batch,
                    kv_head,
                    q_idx,
                    cur_kv,
                    cur_q,
                    sparse_mode,
                )
                if empty:
                    continue
                score = torch.matmul(q_cur.float(), k_sparse.float().T) * scale_value
                probs, x_max, x_sum = softmax(score)
                if fa_param["q_dtype"] == "float16":
                    out = torch.matmul(
                        probs.to(torch.float16).float(), v_sparse.float()
                    )
                elif fa_param["q_dtype"] == "bfloat16":
                    out = torch.matmul(
                        probs.to(torch.bfloat16).float(), v_sparse.float()
                    )
                else:
                    out = torch.matmul(probs.float(), v_sparse.float())
                y[batch, kv_head * group : (kv_head + 1) * group, q_idx, :] = out
                if fa_param.get("return_softmax_lse", False):
                    if fa_param["layout_query"] == "BSND":
                        softmax_max[batch, kv_head, q_idx, :] = x_max[:, 0]
                        softmax_sum[batch, kv_head, q_idx, :] = x_sum[:, 0]
                    else:
                        softmax_max[kv_head, q_offset + q_idx, :] = x_max[:, 0]
                        softmax_sum[kv_head, q_offset + q_idx, :] = x_sum[:, 0]
    return y, softmax_max, softmax_sum


def build_params(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    block_table,
    actual_q,
    actual_kv,
    query_rope,
    key_rope,
    sparse_block_size,
    layout_query,
    layout_kv,
    sparse_mode,
    attention_mode,
    return_softmax_lse,
    block_size,
    block_num,
):
    query_shape = shape_of(query)
    key_shape = shape_of(key)
    sparse_shape = shape_of(sparse_indices)
    block_table_shape = shape_of(block_table, (1,))
    query_rope_shape = shape_of(query_rope, (1,))
    key_rope_shape = shape_of(key_rope, (1,))

    if layout_query == "TND":
        batch = len(actual_q)
        num_heads = query_shape[1]
        total_q = query_shape[0]
        num_kv_heads = key_shape[-2]
        group = num_heads // num_kv_heads
        softmax_shape = (num_kv_heads, total_q, group)
    else:
        batch = query_shape[0]
        seq_q = query_shape[1]
        num_heads = query_shape[2]
        num_kv_heads = key_shape[-2]
        group = num_heads // num_kv_heads
        softmax_shape = (batch, num_kv_heads, seq_q, group)

    params = {
        "actualseqlengths": actual_q,
        "actualseqlengthskv": actual_kv,
        "scalevalue": float(scale_value),
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "sparsemode": int(sparse_mode),
        "sparse_blocksize": int(sparse_block_size),
        "return_softmax_lse": bool(return_softmax_lse),
        "attention_mode": int(attention_mode),
        "shape_input": {
            "query": query_shape,
            "key": key_shape,
            "value": shape_of(value, key_shape),
            "sparse_indices": sparse_shape,
            "block_table": block_table_shape,
            "query_rope": query_rope_shape,
            "key_rope": key_rope_shape,
        },
        "dtype_input": {
            "query": dtype_name(query),
            "key": dtype_name(key),
            "value": dtype_name(value) or dtype_name(key),
            "sparse_indices": dtype_name(sparse_indices) or "int32",
            "block_table": "int32",
            "query_rope": dtype_name(query_rope) or "float16",
            "key_rope": dtype_name(key_rope) or "float16",
        },
        "dtype_output": [dtype_name(query) or "float16"],
        "shape_output": {
            "softmax_max": softmax_shape,
            "softmax_sum": softmax_shape,
        },
    }
    if layout_kv == "PA_BSND":
        params["block_size"] = int(block_size)
        params["block_num"] = int(block_num)
    return params


def compute_with_pytest_golden(input_tensor_dict, params, original_query_dtype):
    pytest_golden = load_pytest_golden_module()
    pytest_golden.gatherKV = ttk_gather_kv
    fa_param = pytest_golden._prepare_fa_param(input_tensor_dict, params)
    if not fa_param["normal_flag"]:
        out = torch.zeros(params["shape_input"]["query"], dtype=original_query_dtype)
        empty = torch.zeros(0, dtype=torch.float32)
        return out, empty, empty

    y_all, softmax_max, softmax_sum = pytest_golden._t_increattention_bnsd(fa_param)
    y_all = pytest_golden.trans_bnsd_to_layout(
        y_all,
        fa_param["out_shape"],
        fa_param["layout_query"],
        fa_param["actualSeqLengths_q"],
    )

    if not params["return_softmax_lse"]:
        empty = torch.zeros(0, dtype=torch.float32)
        return y_all.to(original_query_dtype), empty, empty
    return y_all.to(original_query_dtype), softmax_max, softmax_sum


def cpu_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    *,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
    pre_tokens=(1 << 63) - 1,
    next_tokens=(1 << 63) - 1,
    attention_mode=2,
    return_softmax_lse=False,
    **kwargs,
):
    """Golden reference implementation for sparse flash attention with block-level sparse selection."""
    del (
        pre_tokens,
        next_tokens,
    )  # The pytest CPU formula models mode 0/3 sparse block selection.

    original_query_dtype = query.dtype if torch.is_tensor(query) else torch.float16
    actual_q = to_int_list(actual_seq_lengths_query)
    actual_kv = to_int_list(actual_seq_lengths_kv)

    if query_rope is None and int(attention_mode) == 2:
        query_rope = default_rope_like(query)
    if key_rope is None and int(attention_mode) == 2:
        key_rope = default_rope_like(key)

    if layout_query == "TND" and query is not None:
        actual_q = normalize_tnd_cumulative_seq(actual_q, query.shape[0])
    if layout_kv == "TND" and key is not None:
        actual_kv = normalize_tnd_cumulative_seq(actual_kv, key.shape[0])

    key_for_golden = key
    key_rope_for_golden = key_rope
    # SFA MLA-absorb CPU formula derives V from K; keep the third API tensor
    # for signature compatibility but do not feed it into the CPU math.
    value_for_golden = key_for_golden

    block_size = int(
        kwargs.get(
            "block_size",
            key.shape[1]
            if layout_kv == "PA_BSND" and key is not None and key.dim() > 1
            else 256,
        )
    )
    block_num = int(kwargs.get("block_num", key.shape[0] if key is not None else 1))

    if layout_kv == "PA_BSND" and key is not None and block_table is not None:
        if not actual_kv:
            table = as_cpu(block_table)
            actual_kv = [
                table.shape[1] * block_size
                if table.dim() > 1
                else key.shape[0] * block_size
            ]
        batch = len(actual_kv)
        if key.dim() == 4:
            block_table_cpu = as_cpu(block_table)
            key_for_golden = pa_to_bnsd(
                key, block_table_cpu, actual_kv, batch, block_size
            )
            value_for_golden = key_for_golden
            if key_rope is not None:
                key_rope_for_golden = pa_to_bnsd(
                    key_rope, block_table_cpu, actual_kv, batch, block_size
                )

    if layout_kv == "PA_BSND" and block_table is not None and actual_kv:
        table = as_cpu(block_table)
        capacity = (
            table.shape[1] * block_size if table.dim() > 1 else block_num * block_size
        )
        actual_kv = [min(v, capacity) for v in actual_kv]

    params = build_params(
        query,
        key_for_golden,
        value_for_golden,
        sparse_indices,
        scale_value,
        block_table,
        actual_q,
        actual_kv,
        query_rope,
        key_rope_for_golden,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        attention_mode,
        return_softmax_lse,
        block_size,
        block_num,
    )
    cpu_input = {
        "query": to_cpu_float32_tensor(query),
        "key": to_cpu_float32_tensor(key_for_golden),
        "value": to_cpu_float32_tensor(value_for_golden),
        "sparse_indices": as_cpu(sparse_indices).to(torch.int32)
        if torch.is_tensor(sparse_indices)
        else sparse_indices,
        "block_table": as_cpu(block_table).to(torch.int32)
        if torch.is_tensor(block_table)
        else block_table,
        "query_rope": to_cpu_float32_tensor(query_rope),
        "key_rope": to_cpu_float32_tensor(key_rope_for_golden),
    }
    return compute_with_pytest_golden(cpu_input, params, original_query_dtype)
