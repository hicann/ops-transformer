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

import numpy as np
import torch


PYTEST_GOLDEN_MODULE = None


def load_pytest_golden_module():
    """Load tests/pytest/kv_quant_sparse_flash_attention_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "kv_quant_sparse_flash_attention_golden.py"
    saved_generate_tensor_data = sys.modules.pop("generate_tensor_data", None)
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"qsfa_pytest_golden_{abs(hash(module_path))}", module_path
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
    if params.get("ttk_use_provided_inputs"):
        return compute_golden_with_provided_api_tensors(input_tensor_dict, params)
    return load_pytest_golden_module().compute_golden(input_tensor_dict, params)


def trans_np_hifuint8_tensor_to_float32(in_tensor):
    return load_pytest_golden_module().trans_np_hifuint8_tensor_to_float32(in_tensor)


def softmax(x):
    return load_pytest_golden_module().softmax(x)


# ========================= TTK e2e adapter =========================
# The CPU reference is loaded from tests/pytest; assets keeps only the TTK adapter layer.
# TTK execution consumes the API tensors supplied by the caller instead of regenerating inputs.
__golden__ = {
    "e2e": {
        "torch_npu.npu_kv_quant_sparse_flash_attention": "cpu_kv_quant_sparse_flash_attention"
    }
}


def ttk_tensor_to_list(val):
    if val is None:
        return []
    if torch.is_tensor(val):
        val = val.detach().cpu().tolist()
    elif hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        result = []
        for item in val:
            result.extend(ttk_tensor_to_list(item))
        return result
    return [int(val)]


def ttk_to_cpu_raw(tensor, dtype_hint=None):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.cpu()
    if isinstance(tensor, np.ndarray):
        if tensor.dtype.kind == "V":
            raw_dtype = np.uint8 if tensor.dtype.itemsize == 1 else np.uint16
            raw = np.ascontiguousarray(tensor).view(raw_dtype)
            raw_tensor = torch.from_numpy(raw)
            if tensor.dtype.itemsize == 2:
                return raw_tensor.view(torch.bfloat16)
            if dtype_hint == "float8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
                return raw_tensor.view(torch.float8_e4m3fn)
            return raw_tensor
        return torch.from_numpy(np.ascontiguousarray(tensor))
    return tensor


def ttk_trans_tnd_actseq(seq_list):
    if not seq_list:
        return []
    result = [seq_list[0]]
    for idx in range(len(seq_list) - 1):
        result.append(seq_list[idx + 1] - seq_list[idx])
    return result


def ttk_trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout in ["BSND", "PA_BSND"]:
        return tensor.permute(0, 2, 1, 3).contiguous()
    if layout in ["TND", "TND_NTD"]:
        total = sum(act_q)
        batch = tensor.shape[0]
        head = tensor.shape[1]
        dim = tensor.shape[3]
        output = torch.zeros(size=(total, head, dim), dtype=tensor.dtype)
        t_start = 0
        for batch_idx in range(batch):
            act_s = act_q[batch_idx]
            t_end = t_start + act_s
            if act_s > 0:
                for head_idx in range(head):
                    output[t_start:t_end, head_idx, :] = tensor[
                        batch_idx, head_idx, :act_s, :
                    ]
            t_start += act_s
        return output
    return tensor


def ttk_trans_shape_to_bnsd(tensor, shape, layout, headnums, act_seq):
    if layout in ["BSND", "PA_BSND"]:
        batch = shape[0]
        seq = shape[1]
        head = shape[2]
        dim = shape[3]
        return tensor.permute(0, 2, 1, 3).contiguous(), [batch, head, seq, dim]
    if layout in ["TND", "TND_NTD"]:
        total = shape[0]
        head = shape[1]
        dim = shape[2]
        batch = len(act_seq)
        seq = max(act_seq)
        new_tensor = torch.zeros((batch, head, seq, dim), dtype=tensor.dtype)
        t_start = 0
        for batch_idx in range(batch):
            act_s = act_seq[batch_idx]
            t_end = t_start + act_s
            if act_s > 0:
                for head_idx in range(head):
                    new_tensor[batch_idx, head_idx, 0:act_s, :] = tensor[
                        t_start:t_end, head_idx, :
                    ]
            t_start += act_s
        return new_tensor, [batch, head, seq, dim]
    return tensor, shape


def ttk_pa_to_bnsd(key_pa, block_table, actual_seq_kv, batch, block_size):
    head = key_pa.shape[2]
    dim = key_pa.shape[3]
    max_seq = max(actual_seq_kv) if actual_seq_kv else block_size * block_table.shape[1]
    result = torch.zeros([batch, max_seq, head, dim], dtype=key_pa.dtype)
    for batch_idx in range(batch):
        act_s = actual_seq_kv[batch_idx] if batch_idx < len(actual_seq_kv) else max_seq
        for block_i in range(block_table.shape[1]):
            block_id = int(block_table[batch_idx, block_i].item())
            if block_id == -1:
                continue
            block_offset = block_i * block_size
            block_end = min(block_offset + block_size, act_s)
            if block_offset >= act_s:
                break
            if block_id < 0 or block_id >= key_pa.shape[0]:
                continue
            take = min(block_end - block_offset, key_pa.shape[1])
            if take <= 0:
                continue
            result[batch_idx, block_offset : block_offset + take, :, :] = key_pa[
                block_id, 0:take, :, :
            ]
    return result


def ttk_unpack_packed_kv_cache(kv_packed, d_kv, rope_head_dim, tile_size):
    nope = kv_packed[..., :d_kv]
    rope_bytes = kv_packed[..., d_kv : d_kv + rope_head_dim * 2]
    scale_bytes = kv_packed[..., d_kv + rope_head_dim * 2 :]

    kv_nope = (
        trans_np_hifuint8_tensor_to_float32(nope)
        if nope.dtype == torch.uint8
        else nope.float()
    )
    kv_rope = rope_bytes.contiguous().view(torch.uint8).view(torch.bfloat16).float()
    scale_count = d_kv // tile_size
    kv_dequant_scale = (
        scale_bytes.contiguous().view(torch.uint8).view(torch.float32).float()
    )
    if kv_dequant_scale.shape[-1] != scale_count:
        kv_dequant_scale = kv_dequant_scale.reshape(
            *kv_dequant_scale.shape[:-1], scale_count
        )
    return kv_nope, kv_rope, kv_dequant_scale


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
    if sparse_mode == 0:
        threshold = curr_actual_seq
    elif sparse_mode == 3:
        threshold = curr_actual_seq - curr_actual_seq_q + s1_idx + 1
    else:
        threshold = curr_actual_seq

    threshold = min(threshold, curr_actual_seq)
    if threshold <= 0:
        return True, [], []

    s2_sparse = []
    valid_count = min(sparse_blockcount, math.ceil(threshold / sparse_blocksize))
    for idx in range(valid_count):
        sid = int(sparse_indices[idx].item())
        if sid == -1:
            break
        max_block_id = int((threshold - 1) // sparse_blocksize)
        if sid > max_block_id:
            sid = max_block_id
        begin_idx = sid * sparse_blocksize
        end_idx = min(begin_idx + sparse_blocksize, threshold)
        if begin_idx >= threshold:
            continue
        s2_sparse.extend(range(int(begin_idx), int(end_idx)))

    if len(s2_sparse) == 0:
        return True, [], []
    return (
        False,
        k_tensor[batch, n2_idx, s2_sparse, :],
        v_tensor[batch, n2_idx, s2_sparse, :],
    )


def compute_golden_with_provided_api_tensors(input_tensor_dict, params):
    query = input_tensor_dict.get("query_cache")
    key = input_tensor_dict.get("key_cache")
    value = input_tensor_dict.get("value_cache")
    sparse_indices = input_tensor_dict.get("sparse_indices")
    block_table = input_tensor_dict.get("block_table")

    if query is None or key is None:
        out_shape = list(query.shape) if query is not None else [1, 1, 1, 512]
        out_shape[-1] = 512
        return torch.zeros(out_shape, dtype=torch.float16)

    tensor_dtypes = params.get("tensor_dtypes") or ()
    query_dtype = (
        tensor_dtypes[0] if len(tensor_dtypes) > 0 else params.get("query_dtype")
    )
    key_dtype = params.get(
        "key_dtype", tensor_dtypes[1] if len(tensor_dtypes) > 1 else None
    )
    value_dtype = params.get(
        "value_dtype", tensor_dtypes[2] if len(tensor_dtypes) > 2 else None
    )
    query = ttk_to_cpu_raw(query, query_dtype)
    key = ttk_to_cpu_raw(key, key_dtype)
    value = ttk_to_cpu_raw(value, value_dtype)
    sparse_indices = ttk_to_cpu_raw(sparse_indices)
    block_table = ttk_to_cpu_raw(block_table)

    layout_query = params.get("layout_query", "BSND")
    layout_kv = params.get("layout_kv", "BSND")
    sparse_mode = params.get("sparse_mode", 3)
    sparse_block_size = params.get(
        "sparse_block_size", params.get("sparse_blocksize", 1)
    )
    tile_size = params.get("tile_size", 128)
    rope_head_dim = params.get("rope_head_dim", 64)
    scale_value = params.get("scale_value", params.get("scalevalue", 1.0))

    act_q = ttk_tensor_to_list(params.get("actual_seq_lengths_query"))
    act_kv = ttk_tensor_to_list(params.get("actual_seq_lengths_kv"))

    is_tnd_q = layout_query in ["TND"]
    is_tnd_kv = layout_kv in ["TND"]
    is_pa = layout_kv in ["PA_BSND"]
    act_q_per_batch = ttk_trans_tnd_actseq(act_q) if is_tnd_q and act_q else act_q
    act_kv_per_batch = ttk_trans_tnd_actseq(act_kv) if is_tnd_kv and act_kv else act_kv

    q_shape = list(query.shape)
    k_shape = list(key.shape)
    if is_tnd_q:
        batch = len(act_q_per_batch) if act_q_per_batch else 1
        num_heads = q_shape[1]
        seq_q = max(act_q_per_batch) if act_q_per_batch else q_shape[0]
    else:
        batch = q_shape[0]
        seq_q = q_shape[1]
        num_heads = q_shape[2]

    num_kv_heads = k_shape[1] if is_tnd_kv else k_shape[2]
    d_kv = 512
    d_full = d_kv + rope_head_dim

    if not act_q_per_batch:
        act_q_per_batch = [q_shape[0]] if is_tnd_q else [seq_q] * batch
    if not act_kv_per_batch:
        if is_tnd_kv:
            act_kv_per_batch = [k_shape[0]]
        elif is_pa:
            act_kv_per_batch = [1024] * batch
        else:
            act_kv_per_batch = [k_shape[1]] * batch
    if len(act_q_per_batch) < batch:
        act_q_per_batch = [act_q_per_batch[0]] * batch
    if len(act_kv_per_batch) < batch:
        act_kv_per_batch = [act_kv_per_batch[0]] * batch

    q_nope = query[..., :d_kv].cpu().float()
    q_rope = query[..., d_kv : d_kv + rope_head_dim].cpu().float()
    q_full = torch.cat([q_nope, q_rope], dim=-1)
    q_full_shape = [q_shape[0], q_shape[1], d_full] if is_tnd_q else list(q_shape)
    q_full_shape[-1] = d_full
    q_bnsd, q_bnsd_shape = ttk_trans_shape_to_bnsd(
        q_full, q_full_shape, layout_query, num_heads, act_q_per_batch
    )

    k_raw = ttk_to_cpu_raw(key)
    if is_pa:
        block_table_cpu = ttk_to_cpu_raw(block_table)
        block_size_val = int(
            params.get("block_size", k_shape[1] if len(k_shape) > 1 else tile_size)
        )
        if block_table_cpu is not None:
            k_bsnd = ttk_pa_to_bnsd(
                k_raw, block_table_cpu, act_kv_per_batch, batch, block_size_val
            )
        else:
            k_bsnd = k_raw.float()
    else:
        k_bsnd = k_raw

    k_nope, k_rope, k_scale = ttk_unpack_packed_kv_cache(
        k_bsnd, d_kv, rope_head_dim, tile_size
    )
    k_full = torch.cat([k_nope, k_rope], dim=-1)

    if is_pa:
        effective_kv_layout = "BSND"
        kv_shape_for_bnsd = [batch, k_full.shape[1], k_full.shape[2], d_full]
    elif is_tnd_kv:
        effective_kv_layout = layout_kv
        kv_shape_for_bnsd = [k_bsnd.shape[0], k_bsnd.shape[1], d_full]
    else:
        effective_kv_layout = layout_kv
        kv_shape_for_bnsd = list(k_bsnd.shape)
        kv_shape_for_bnsd[-1] = d_full

    k_bnsd, k_bnsd_shape = ttk_trans_shape_to_bnsd(
        k_full, kv_shape_for_bnsd, effective_kv_layout, num_kv_heads, act_kv_per_batch
    )
    k_scale_bnsd, _ = ttk_trans_shape_to_bnsd(
        k_scale,
        list(k_scale.shape),
        effective_kv_layout,
        num_kv_heads,
        act_kv_per_batch,
    )

    k_bnsd_float = k_bnsd.float()
    k_scale_bnsd_float = k_scale_bnsd.float()
    k_bnsd_float[..., :d_kv] = k_bnsd_float[..., :d_kv] * torch.repeat_interleave(
        k_scale_bnsd_float, tile_size, dim=-1
    )
    # MLA-absorb QSFA tests use the dequantized K_nope slice as V.
    v_bnsd_float = k_bnsd_float[..., :d_kv].clone()

    si = ttk_to_cpu_raw(sparse_indices)
    if si is not None:
        si_bnsd, si_bnsd_shape = ttk_trans_shape_to_bnsd(
            si, list(si.shape), layout_query, num_kv_heads, act_q_per_batch
        )
    else:
        si_bnsd = torch.zeros((batch, num_kv_heads, seq_q, 1), dtype=torch.int32)
        si_bnsd_shape = list(si_bnsd.shape)
    sparse_blockcount = si_bnsd_shape[-1]

    out_shape_bnsd = list(q_bnsd_shape)
    out_shape_bnsd[-1] = d_kv
    y = torch.zeros(out_shape_bnsd, dtype=torch.float32)
    group = num_heads // num_kv_heads

    for batch_idx in range(batch):
        batch_q_len = (
            int(act_q_per_batch[batch_idx])
            if batch_idx < len(act_q_per_batch)
            else seq_q
        )
        batch_kv_len = (
            int(act_kv_per_batch[batch_idx])
            if batch_idx < len(act_kv_per_batch)
            else k_bnsd_shape[2]
        )
        for n2_idx in range(num_kv_heads):
            for s1_idx in range(batch_q_len):
                if s1_idx < batch_q_len - batch_kv_len and sparse_mode != 0:
                    continue
                q_curr = q_bnsd[
                    batch_idx, n2_idx * group : (n2_idx + 1) * group, s1_idx, :
                ]
                si_curr = si_bnsd[batch_idx, n2_idx, s1_idx, :]
                empty, k_sparse, v_sparse = ttk_gather_kv(
                    k_bnsd_float,
                    v_bnsd_float,
                    si_curr,
                    sparse_block_size,
                    sparse_blockcount,
                    batch_idx,
                    n2_idx,
                    s1_idx,
                    batch_kv_len,
                    batch_q_len,
                    sparse_mode,
                )
                if empty:
                    continue
                bmm1 = torch.matmul(q_curr.float(), k_sparse.float().T)
                softmax_res = softmax(bmm1 * scale_value)
                bmm2 = torch.matmul(softmax_res.float(), v_sparse.float())
                y[batch_idx, n2_idx * group : (n2_idx + 1) * group, s1_idx, :] = bmm2

    out_shape = list(q_nope.shape)
    y_out = ttk_trans_bnsd_to_layout(y, out_shape, layout_query, act_q_per_batch)
    return y_out.to(query.dtype)


def cpu_kv_quant_sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    *,
    key_dequant_scale=None,
    value_dequant_scale=None,
    block_table=None,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=3,
    pre_tokens=(1 << 63) - 1,
    next_tokens=(1 << 63) - 1,
    attention_mode=0,
    quant_scale_repo_mode=1,
    tile_size=128,
    rope_head_dim=64,
    key_dtype=None,
    value_dtype=None,
    **kwargs,
):
    """Golden reference implementation for KV-quantized sparse flash attention with MLA-absorb support."""
    input_tensor_dict = {
        "query_cache": query,
        "key_cache": key,
        "value_cache": value,
        "sparse_indices": sparse_indices,
        "key_dequant_scale": key_dequant_scale,
        "value_dequant_scale": value_dequant_scale,
        "block_table": block_table,
    }
    params = {
        "ttk_use_provided_inputs": True,
        "scale_value": scale_value,
        "scalevalue": scale_value,
        "key_quant_mode": key_quant_mode,
        "value_quant_mode": value_quant_mode,
        "key_dequant_scale": key_dequant_scale,
        "value_dequant_scale": value_dequant_scale,
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "sparse_block_size": sparse_block_size,
        "sparse_blocksize": sparse_block_size,
        "layout_query": layout_query,
        "layout_kv": layout_kv,
        "sparse_mode": sparse_mode,
        "pre_tokens": pre_tokens,
        "next_tokens": next_tokens,
        "attention_mode": attention_mode,
        "quant_scale_repo_mode": quant_scale_repo_mode,
        "tile_size": tile_size,
        "rope_head_dim": rope_head_dim,
        "key_dtype": key_dtype,
        "value_dtype": value_dtype,
    }
    params.update(kwargs)
    return compute_golden(input_tensor_dict, params)
