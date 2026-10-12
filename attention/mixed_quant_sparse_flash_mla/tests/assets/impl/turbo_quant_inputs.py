# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU input preparation for the A2/A3 TurboQuant CSA contract."""

import math

import numpy as np
import torch


def _floating(shape, dtype, bounds):
    low, high = (float(value) for value in (bounds or (-1, 1)))
    if not math.isfinite(low) or not math.isfinite(high):
        if low != high and not (math.isnan(low) and math.isnan(high)):
            raise ValueError("non-finite data ranges must have equal endpoints")
        values = np.full(shape, low, dtype=np.float32)
    else:
        if low > high:
            raise ValueError("input data ranges must be ordered")
        values = np.random.uniform(low, high, shape).astype(np.float32)
    return torch.from_numpy(values).to(dtype)


def _block_table(shape, block_size, lengths, block_count):
    table = torch.full(shape, -1, dtype=torch.int32)
    needed = [math.ceil(value / block_size) for value in lengths]
    if sum(needed) > block_count or max(needed, default=0) > shape[1]:
        raise ValueError("TurboQuant block table/cache capacity is too small")
    pages = np.random.permutation(block_count)
    offset = 0
    for batch, count in enumerate(needed):
        table[batch, :count] = torch.as_tensor(
            pages[offset : offset + count], dtype=torch.int32
        )
        offset += count
    return table


def generate_case(
    params,
    q,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    ori_block_table,
    cmp_block_table,
    sinks,
):
    """Generate packed slots, logical causal indices and matching PA metadata."""
    if params.get("layout_q") != "TND" or params.get("layout_kv") != "PA_BBND":
        raise ValueError("TurboQuant requires TND/PA_BBND")
    if params.get("template_run_mode", "CSA") != "CSA":
        raise ValueError("TurboQuant supports only CSA")
    if params.get("batch_consistency") or all(
        params.get(name) is not None
        for name in ("batch_axis", "batch_slice_info", "batch_seed")
    ):
        raise ValueError(
            "TurboQuant CPU input preparation does not support batch relation generation"
        )
    required = (ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table, cmp_block_table)
    if any(value is None for value in required):
        raise ValueError(
            "TurboQuant requires both KV tensors, CMP indices and both block tables"
        )
    if (
        q.ndim != 3
        or q.shape[-1] != 512
        or not (4 <= q.shape[1] <= 128)
        or q.shape[1] % 4
    ):
        raise ValueError(
            "TurboQuant q must have shape (T, N, 512), N in [4, 128] aligned to 4"
        )
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or ori_kv.dtype != q.dtype
        or cmp_kv.dtype != torch.uint8
    ):
        raise ValueError(
            "TurboQuant requires matching FP16/BF16 q/ori_kv and uint8 cmp_kv"
        )
    for tensor, width in ((ori_kv, 512), (cmp_kv, 258)):
        if tensor.ndim != 4 or tuple(tensor.shape[2:]) != (1, width):
            raise ValueError(
                "TurboQuant KV must have shape (blocks, block_size, 1, 512/258)"
            )
        if (
            tensor.shape[0] < 1
            or not (16 <= tensor.shape[1] <= 1024)
            or tensor.shape[1] % 16
        ):
            raise ValueError(
                "TurboQuant cache blocks must be nonempty and aligned to 16"
            )
    ratio = int(params.get("cmp_ratio", 4))
    if (
        ratio not in (4, 128)
        or cmp_sparse_indices.ndim != 3
        or tuple(cmp_sparse_indices.shape[:2]) != (q.shape[0], 1)
        or cmp_sparse_indices.shape[-1] not in (512, 1024)
        or cmp_sparse_indices.dtype != torch.int32
    ):
        raise ValueError(
            "TurboQuant requires cmp_ratio=4/128 and CMP indices shape (T, 1, 512/1024)"
        )
    if sinks is not None and (
        tuple(sinks.shape) != (q.shape[1],) or sinks.dtype != torch.float32
    ):
        raise ValueError("TurboQuant sinks must have shape (N,) and dtype float32")
    contract = tuple(
        int(params.get(name, default))
        for name, default in (
            ("ori_mask_mode", 4),
            ("cmp_mask_mode", 3),
            ("ori_win_right", 0),
            ("rope_head_dim", 64),
            ("topk_value_mode", 1),
        )
    )
    if contract != (4, 3, 0, 64, 1) or int(params.get("ori_win_left", 127)) < 0:
        raise ValueError(
            "TurboQuant requires masks=4/3, nonnegative left window, right=0, RoPE=64 and logical indices"
        )
    cuq = params.get("cu_seqlens_q")
    olen = params.get("seqused_ori_kv")
    if cuq is None or olen is None:
        raise ValueError("TurboQuant requires cu_seqlens_q and seqused_ori_kv values")
    cuq, olen = [int(value) for value in cuq], [int(value) for value in olen]
    qlen = [right - left for left, right in zip(cuq, cuq[1:])]
    if not olen or len(cuq) != len(olen) + 1 or cuq[0] != 0 or cuq[-1] != q.shape[0]:
        raise ValueError(
            "TurboQuant query prefix must start at zero, contain B+1 values and end at T"
        )
    if any(qs < 0 or os < qs for qs, os in zip(qlen, olen)):
        raise ValueError(
            "TurboQuant requires nondecreasing prefixes and ORI lengths >= query lengths"
        )
    if (
        ori_block_table.ndim != 2
        or cmp_block_table.ndim != 2
        or ori_block_table.dtype != torch.int32
        or cmp_block_table.dtype != torch.int32
        or ori_block_table.shape[0] != len(olen)
        or cmp_block_table.shape[0] != len(olen)
    ):
        raise ValueError("TurboQuant block tables must have B rows")
    clen = [value // ratio for value in olen]
    op_input = {
        name: None
        for name in (
            "ori_sparse_indices",
            "cu_seqlens_ori_kv",
            "cu_seqlens_cmp_kv",
            "seqused_q",
            "seqused_cmp_kv",
            "cmp_residual_kv",
            "ori_topk_length",
            "cmp_topk_length",
        )
    }
    op_input["q"] = _floating(tuple(q.shape), q.dtype, params.get("q_datarange"))
    op_input["ori_kv"] = _floating(
        tuple(ori_kv.shape), ori_kv.dtype, params.get("ori_kv_datarange")
    )
    packed = np.random.randint(0, 256, tuple(cmp_kv.shape), dtype=np.uint8)
    # Each nibble indexes a TQ4 centroid; the final two bytes hold a little-endian FP16 scale.
    scale_bounds = params.get("tq_scale_range", (0.125, 2.0))
    low, high = [float(value) for value in scale_bounds]
    if not (0 < low <= high <= np.finfo(np.float16).max) or low < np.nextafter(
        np.float16(0), np.float16(1)
    ):
        raise ValueError(
            "tq_scale_range must remain finite and strictly positive in FP16"
        )
    scales = np.random.uniform(low, high, packed.shape[:-1]).astype("<f2")
    packed[..., 256:] = scales[..., None].view(np.uint8)
    op_input["cmp_kv"] = torch.from_numpy(packed)
    sparse = torch.full(tuple(cmp_sparse_indices.shape), -1, dtype=torch.int32)
    for batch, qs in enumerate(qlen):
        for position in range(qs):
            causal = (olen[batch] - qs + position + 1) // ratio
            count = min(causal, sparse.shape[-1])
            selected = np.sort(np.random.permutation(causal)[:count])
            sparse[cuq[batch] + position, 0, :count] = torch.as_tensor(
                selected, dtype=torch.int32
            )
    op_input["cmp_sparse_indices"] = sparse
    op_input["ori_block_table"] = _block_table(
        tuple(ori_block_table.shape), ori_kv.shape[1], olen, ori_kv.shape[0]
    )
    op_input["cmp_block_table"] = _block_table(
        tuple(cmp_block_table.shape), cmp_kv.shape[1], clen, cmp_kv.shape[0]
    )
    op_input["cu_seqlens_q"] = torch.tensor(cuq, dtype=torch.int32)
    op_input["seqused_ori_kv"] = torch.tensor(olen, dtype=torch.int32)
    op_input["sinks"] = (
        _floating(tuple(sinks.shape), torch.float32, (-1, 1))
        if sinks is not None
        else None
    )
    metadata = dict(
        num_heads_q=int(q.shape[1]),
        num_heads_kv=1,
        head_dim=512,
        batch_size=len(olen),
        max_seqlen_q=max(qlen),
        max_seqlen_ori_kv=max(olen),
        max_seqlen_cmp_kv=max(clen),
        topk=int(sparse.shape[-1]),
        ori_topk=0,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=int(params.get("ori_win_left", 127)),
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=True,
        quant_mode=3,
        rope_head_dim=64,
    )
    for name in (
        "cu_seqlens_q",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "seqused_ori_kv",
        "seqused_cmp_kv",
        "cmp_residual_kv",
        "ori_topk_length",
        "cmp_topk_length",
    ):
        metadata[name] = op_input.get(name)
    return dict(
        Testcase_Name=params.get("testcase_name"),
        params=params,
        op_input=op_input,
        metadata_input=metadata,
        cpu_output=None,
        cpu_lse=None,
    )
