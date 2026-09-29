# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import torch

from qsli_parameter_normalization import normalize_qsli_params
from quant_sparse_lightning_indexer_golden import (
    pack_candidate_k,
    quant_sparse_lightning_indexer_golden,
    quant_sparse_lightning_indexer_golden_selected,
)

_compare_spec = importlib.util.spec_from_file_location(
    "qsli_result_compare_method", Path(__file__).with_name("result_compare_method.py")
)
result_compare_method = importlib.util.module_from_spec(_compare_spec)
_compare_spec.loader.exec_module(result_compare_method)

BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
N1 = 32
N2 = 1
PACKED_D = 64
FP4_E2M1_VALUES = torch.tensor(
    (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ),
    dtype=torch.float32,
)
E8M0_ONE_VALUE = 127


def _range_pair(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range, got {value!r}")
    low, high = float(value[0]), float(value[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"invalid {name}: {value!r}")
    return low, high


def _make_mxfp4_bytes(logical_shape, data_range, generator):
    low, high = _range_pair(data_range, "FP4 E2M1 numeric range")
    if logical_shape[-1] % 2:
        raise ValueError(f"MXFP4 head_dim must be even, got {logical_shape[-1]}")
    valid_codes = (
        torch.nonzero(
            (FP4_E2M1_VALUES >= low) & (FP4_E2M1_VALUES <= high),
            as_tuple=False,
        )
        .flatten()
        .to(torch.uint8)
    )
    if valid_codes.numel() == 0:
        raise ValueError(
            f"FP4 E2M1 range contains no representable value: {data_range}"
        )
    raw = valid_codes[
        torch.randint(
            valid_codes.numel(),
            tuple(logical_shape),
            generator=generator,
            dtype=torch.long,
        )
    ]
    raw_rows = raw.reshape(-1, logical_shape[-1])
    coverage_count = min(valid_codes.numel(), logical_shape[-1])
    coverage = torch.arange(coverage_count).unsqueeze(0) + torch.arange(
        raw_rows.shape[0]
    ).unsqueeze(1)
    raw_rows[:, :coverage_count] = valid_codes[coverage % valid_codes.numel()]
    return (raw[..., 0::2] | (raw[..., 1::2] << 4)).contiguous()


def _make_e8m0_bytes(shape, scale_range):
    low, high = _range_pair(scale_range, "E8M0 scale range")
    if low <= 0:
        raise ValueError(f"E8M0 scale range must be positive, got {scale_range}")
    min_code = max(0, math.ceil(math.log2(low)) + E8M0_ONE_VALUE)
    max_code = min(254, math.floor(math.log2(high)) + E8M0_ONE_VALUE)
    while min_code <= 254 and math.ldexp(1.0, min_code - E8M0_ONE_VALUE) < low:
        min_code += 1
    while max_code >= 0 and math.ldexp(1.0, max_code - E8M0_ONE_VALUE) > high:
        max_code -= 1
    if min_code > max_code:
        raise ValueError(
            f"E8M0 scale range contains no representable value: {scale_range}"
        )
    count = max_code - min_code + 1
    return (
        torch.arange(math.prod(shape), dtype=torch.int64)
        .remainder(count)
        .add(min_code)
        .reshape(shape)
        .to(torch.uint8)
    )


def _make_dim0_view(tensor, padding):
    rows = tensor.shape[0]
    flat = tensor.reshape(rows, -1)
    storage = torch.zeros((rows, flat.shape[1] + padding), dtype=tensor.dtype)
    storage[:, : flat.shape[1]] = flat
    return storage[:, : flat.shape[1]].view(tensor.shape), storage


def _int_tensor(value, default):
    source = default if value is None else value
    return torch.tensor(source, dtype=torch.int32)


def _is_auto(value):
    return isinstance(value, str) and value.upper() == "AUTO"


def _make_candidate_inputs(p, rows, logical_blocks, generator, cu):
    explicit_indices = p["candidate_block_indices"]
    explicit_lengths = p["candidate_block_length"]
    candidate = torch.full((rows, N2, CANDIDATE_CAPACITY), -1, dtype=torch.int32)
    auto_indices = _is_auto(explicit_indices)
    if auto_indices:
        auto_lengths = explicit_lengths is None or _is_auto(explicit_lengths)
        requested_lengths = None
        scalar_length = None
        if not auto_lengths:
            requested_lengths = torch.tensor(explicit_lengths, dtype=torch.int32)
            if requested_lengths.ndim == 0:
                scalar_length = int(requested_lengths)
                requested_lengths = None
            elif tuple(requested_lengths.shape) != (rows, N2):
                raise ValueError(
                    "candidate_block_length from Excel must have shape "
                    f"(T1,N2), got {tuple(requested_lengths.shape)}"
                )

        used_k_value = p["seqused_k"]
        used_k = _int_tensor(
            [p["sequence_length"]] * p["batch_size"]
            if used_k_value is None or _is_auto(used_k_value)
            else used_k_value,
            [p["sequence_length"]] * p["batch_size"],
        )
        if tuple(used_k.shape) != (p["batch_size"],):
            raise ValueError("seqused_k must have shape (B,)")
        lengths = torch.empty((rows, N2), dtype=torch.int32)
        for batch_index in range(p["batch_size"]):
            begin = int(cu[batch_index])
            end = int(cu[batch_index + 1])
            batch_blocks = min(
                CANDIDATE_CAPACITY,
                (int(used_k[batch_index]) + BLOCK_SIZE - 1) // BLOCK_SIZE,
            )
            if batch_blocks < 0 or batch_blocks > logical_blocks:
                raise ValueError("seqused_k must be within [0, k_seq]")
            if requested_lengths is not None:
                batch_lengths = requested_lengths[begin:end, 0]
                if torch.any(batch_lengths < 0) or torch.any(
                    batch_lengths > batch_blocks
                ):
                    raise ValueError(
                        "candidate_block_length must be within the valid block count "
                        "of its batch"
                    )
            else:
                requested = batch_blocks if scalar_length is None else scalar_length
                if requested < 0:
                    raise ValueError("candidate_block_length must be non-negative")
                batch_lengths = torch.full(
                    (end - begin,), min(requested, batch_blocks), dtype=torch.int32
                )
            lengths[begin:end, 0] = batch_lengths
            max_length = int(batch_lengths.max()) if end > begin else 0
            order = torch.randperm(batch_blocks, generator=generator)[:max_length]
            for row_index in range(begin, end):
                row_length = int(lengths[row_index, 0])
                candidate[row_index, 0, :row_length] = order[:row_length].to(
                    torch.int32
                )
        return candidate, lengths

    source = torch.tensor(explicit_indices, dtype=torch.int32)
    if source.ndim != 3 or tuple(source.shape[:2]) != (rows, N2):
        raise ValueError(
            "candidate_block_indices from Excel must have shape "
            f"(T1,N2,L), got {tuple(source.shape)}"
        )
    if source.shape[2] > CANDIDATE_CAPACITY:
        raise ValueError(f"candidate_block_indices L must be <= {CANDIDATE_CAPACITY}")
    if explicit_lengths is None:
        raise ValueError(
            "candidate_block_length is required when candidate_block_indices is explicit"
        )
    lengths = torch.tensor(explicit_lengths, dtype=torch.int32)
    if tuple(lengths.shape) != (rows, N2):
        raise ValueError(
            "candidate_block_length from Excel must have shape "
            f"(T1,N2), got {tuple(lengths.shape)}"
        )
    candidate[:, :, : source.shape[2]] = source
    for row_index in range(rows):
        for head_index in range(N2):
            length = int(lengths[row_index, head_index])
            if length < 0 or length > source.shape[2]:
                raise ValueError(
                    "candidate_block_length must be within the explicit candidate width"
                )
            valid = source[row_index, head_index, :length]
            if torch.any(valid < 0) or torch.any(valid >= logical_blocks):
                raise ValueError(
                    "candidate_block_indices contains an out-of-range logical block"
                )
    return candidate, lengths


def generate_qsli_test_data(params, *, defer_golden=False):
    p = normalize_qsli_params(params)
    batch, q_len, s2 = p["batch_size"], p["query_length"], p["sequence_length"]
    if s2 < 1 or q_len < 1:
        raise ValueError("query_length and sequence_length must be positive")
    if p["qk_dtype"] not in (
        "FLOAT4_E2M1",
        "FLOAT4_E2M1FN_X2",
        "torch.float4_e2m1fn_x2",
    ):
        raise ValueError(f"QSLI requires MXFP4 qk_dtype, got {p['qk_dtype']}")
    if p["dequant_dtype"] not in (
        "FLOAT8_E8M0",
        "FLOAT8_E8M0FNU",
        "torch.float8_e8m0fnu",
    ):
        raise ValueError(f"QSLI requires E8M0 dequant_dtype, got {p['dequant_dtype']}")
    if p["weight_dtype"] not in ("FP32", "FLOAT32", "torch.float32"):
        raise ValueError(f"QSLI requires FP32 weight_dtype, got {p['weight_dtype']}")
    if p["actual_seq_dtype"] not in ("INT32", "torch.int32"):
        raise ValueError(
            f"QSLI requires INT32 actual_seq_dtype, got {p['actual_seq_dtype']}"
        )
    if p["quant_mode"] != 1:
        raise ValueError("QSLI DSL public contract requires quant_mode=1")
    if p["candidate_block_size"] != BLOCK_SIZE:
        raise ValueError(f"QSLI candidate_block_size must be {BLOCK_SIZE}")
    if p["layout_q"] != "TND" or p["layout_k"] not in ("PA_BBND", "TND"):
        raise ValueError("QSLI requires TND queries and PA_BBND or TND keys")
    if p["q_t_size"] < 1:
        raise ValueError("q_t_size must be positive")
    if p["k_t_size"] != p["sequence_length"]:
        raise ValueError("k_t_size must equal k_seq")
    if (p["q_head_num"], p["k_head_num"], p["head_dim"]) != (N1, N2, 128):
        raise ValueError("QSLI DSL requires q_head_num=32, k_head_num=1, head_dim=128")
    page_size = int(p["block_size"])
    if page_size <= 0 or page_size % BLOCK_SIZE:
        raise ValueError("block_size must be a positive multiple of 8")
    generator = torch.Generator(device="cpu").manual_seed(p["seed"])
    rows = p["q_t_size"]
    if p["cu_seqlens_q"] is None and (batch != 1 or rows != q_len):
        raise ValueError(
            "cu_seqlens_q=None is only valid for batch_size=1 and q_t_size=q_seq"
        )
    default_cu = [index * q_len for index in range(batch + 1)]
    cu_value = p["cu_seqlens_q"]
    cu = (
        _int_tensor(default_cu, default_cu)
        if _is_auto(cu_value)
        else None
        if cu_value is None
        else _int_tensor(cu_value, cu_value)
    )
    if cu is not None and (
        tuple(cu.shape) != (batch + 1,) or int(cu[0]) != 0 or int(cu[-1]) != rows
    ):
        raise ValueError(
            "cu_seqlens_q must have shape (B+1), start at 0, and end at q_t_size"
        )
    q_lengths = [rows] if cu is None else (cu[1:] - cu[:-1]).tolist()
    if any(length < 0 or length > q_len for length in q_lengths):
        raise ValueError("each TND query length must be within [0, q_seq]")
    used_q_value = p["seqused_q"]
    used_q = (
        _int_tensor(q_lengths, q_lengths)
        if _is_auto(used_q_value)
        else None
        if used_q_value is None
        else _int_tensor(used_q_value, used_q_value)
    )
    if used_q is not None and (
        tuple(used_q.shape) != (batch,)
        or any(
            int(value) < 0 or int(value) > int(q_lengths[index])
            for index, value in enumerate(used_q.tolist())
        )
    ):
        raise ValueError("seqused_q must have shape (B,) and fit cu_seqlens_q")
    pages = (s2 + page_size - 1) // page_size
    if p["layout_k"] == "PA_BBND" and p["block_num"] != pages:
        raise ValueError(f"block_num must equal ceil(k_seq/block_size)={pages}")
    cu_k = _int_tensor(p["cu_seqlens_k"], list(range(0, batch * s2 + 1, s2)))
    if tuple(cu_k.shape) != (batch + 1,) or int(cu_k[0]) != 0:
        raise ValueError("cu_seqlens_k must have shape (B+1,) and start at zero")
    logical_blocks = (s2 + BLOCK_SIZE - 1) // BLOCK_SIZE

    q = _make_mxfp4_bytes((rows, N1, PACKED_D * 2), p["query_datarange"], generator)
    k_shape = (
        (int(cu_k[-1]), N2, PACKED_D * 2)
        if p["layout_k"] == "TND"
        else (pages, page_size, N2, PACKED_D * 2)
    )
    k = _make_mxfp4_bytes(k_shape, p["key_datarange"], generator)
    descale_q = _make_e8m0_bytes((rows, N1, 2, 2), p["q_scale_datarange"])
    scale_shape = (
        (int(cu_k[-1]), N2, 2, 2)
        if p["layout_k"] == "TND"
        else (pages, page_size, N2, 2, 2)
    )
    descale_k = _make_e8m0_bytes(scale_shape, p["k_scale_datarange"])
    weight_low, weight_high = _range_pair(p["weights_datarange"], "weights_datarange")
    weights = torch.rand((rows, N1), generator=generator, dtype=torch.float32)
    weights = weights * (weight_high - weight_low) + weight_low
    k_storage = scale_storage = fused_storage = None
    if p["storage_mode"] == "dim0_noncontiguous" and p["layout_k"] == "PA_BBND":
        key_size = page_size * N2 * PACKED_D
        scale_size = page_size * N2 * 4
        fused_storage = torch.empty((pages, key_size + scale_size), dtype=torch.uint8)
        fused_storage[:, :key_size] = k.reshape(pages, -1)
        fused_storage[:, key_size:] = descale_k.reshape(pages, -1)
        k = fused_storage[:, :key_size].view_as(k)
        descale_k = fused_storage[:, key_size:].view_as(descale_k)
    elif p["storage_mode"] == "dim0_noncontiguous":
        k, k_storage = _make_dim0_view(k, 64)
        descale_k, scale_storage = _make_dim0_view(descale_k, 32)
    elif p["storage_mode"] == "key_dim0_noncontiguous":
        k, k_storage = _make_dim0_view(k, 64)
    elif p["storage_mode"] == "scale_dim0_noncontiguous":
        descale_k, scale_storage = _make_dim0_view(descale_k, 32)
    elif p["storage_mode"] != "contiguous":
        raise ValueError(f"unsupported storage_mode: {p['storage_mode']}")

    candidate_cu = _int_tensor(default_cu, default_cu) if cu is None else cu
    candidate, candidate_length = _make_candidate_inputs(
        p, rows, logical_blocks, generator, candidate_cu
    )
    auto_block_table = _is_auto(p["block_table"])
    block_table = (
        None
        if p["layout_k"] == "TND"
        else (
            torch.stack(
                [
                    torch.randperm(pages, generator=generator).to(torch.int32)
                    for _ in range(batch)
                ]
            )
            if auto_block_table
            else None
            if p["block_table"] is None
            else _int_tensor(p["block_table"], p["block_table"])
        )
    )
    if block_table is not None and tuple(block_table.shape) != (batch, pages):
        raise ValueError(
            f"block_table must have shape ({batch},{pages}), got {tuple(block_table.shape)}"
        )
    used_k_value = p["seqused_k"]
    used_k = (
        _int_tensor([s2] * batch, [s2] * batch)
        if _is_auto(used_k_value)
        else None
        if used_k_value is None
        else _int_tensor(used_k_value, used_k_value)
    )
    if used_k is not None and (
        tuple(used_k.shape) != (batch,)
        or any(int(value) < 0 or int(value) > s2 for value in used_k.tolist())
    ):
        raise ValueError("seqused_k must have shape (B,) and values in [0,k_seq]")
    residual_value = p["cmp_residual_k"]
    if _is_auto(residual_value):
        residual_value = [index % p["cmp_ratio"] for index in range(batch)]
    residual = (
        None if residual_value is None else _int_tensor(residual_value, residual_value)
    )
    offset_value = p["output_idx_offset"]
    auto_offset = _is_auto(offset_value)
    if isinstance(offset_value, bool) or auto_offset:
        offset = (
            torch.arange(rows, dtype=torch.int32).reshape(rows, 1) * 3 + 37
            if auto_offset or offset_value
            else None
        )
    elif offset_value is None:
        offset = None
    else:
        offset = _int_tensor(offset_value, offset_value)
        if tuple(offset.shape) != (rows, N2):
            raise ValueError(
                f"output_idx_offset must have shape ({rows},{N2}), got {tuple(offset.shape)}"
            )
    tensors = {
        "q": q,
        "k": k,
        "weights": weights,
        "descale_q": descale_q,
        "descale_k": descale_k,
        "candidate": candidate,
        "candidate_length": candidate_length,
        "cu": cu,
        "cu_k": cu_k,
        "used_q": used_q,
        "used_k": used_k,
        "residual": residual,
        "block_table": block_table,
        "offset": offset,
        "_fused_storage": fused_storage,
        "_k_storage": k_storage,
        "_scale_storage": scale_storage,
    }
    golden_key = pack_candidate_k(k, descale_k) if p["layout_k"] == "PA_BBND" else k
    golden_kwargs = dict(
        descale_k=descale_k if p["layout_k"] == "TND" else None,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu_k,
        seqused_q=tensors["used_q"],
        seqused_k=tensors["used_k"],
        cmp_residual_k=tensors["residual"],
        block_table=block_table,
        output_idx_offset=offset,
        topk=p["topk"],
        candidate_block_size=p["candidate_block_size"],
        quant_mode=p["quant_mode"],
        max_seqlen_q=p["max_seqlen_q"],
        mask_mode=p["mask_mode"],
        cmp_ratio=p["cmp_ratio"],
        layout_q=p["layout_q"],
        layout_k=p["layout_k"],
        return_value=p["return_value"],
    )
    selected_rows = None
    if defer_golden:
        golden = None
    elif q_len >= 8192:
        selected_rows = []
        active_q = q_lengths if used_q is None else used_q.tolist()
        boundaries = candidate_cu.tolist()
        for batch_index, active in enumerate(active_q):
            if active <= 0:
                continue
            begin = boundaries[batch_index]
            selected_rows.extend((begin, begin + active // 2, begin + active - 1))
        selected_rows = sorted(set(selected_rows))
        golden = quant_sparse_lightning_indexer_golden_selected(
            q,
            golden_key,
            tensors["weights"],
            tensors["descale_q"],
            candidate,
            candidate_length,
            selected_rows,
            **golden_kwargs,
        )
    else:
        golden = quant_sparse_lightning_indexer_golden(
            q,
            golden_key,
            tensors["weights"],
            tensors["descale_q"],
            candidate,
            candidate_length,
            **golden_kwargs,
        )
    tensors["_selected_rows"] = selected_rows
    if p["layout_k"] == "PA_BBND":
        packed = golden_key
        if p["storage_mode"] != "contiguous":
            packed, backing = _make_dim0_view(packed, 32)
            tensors["_fused_storage"] = backing
        else:
            tensors["_fused_storage"] = None
        tensors["k"] = packed
        tensors.pop("descale_k")
        tensors["_k_storage"] = None
        tensors["_scale_storage"] = None
    return {"params": p, "tensors": tensors, "cpu_result": golden}


def convert_qsli_case_to_tnd(case_data):
    params = dict(case_data["params"])
    if params["layout_k"] != "PA_BBND":
        raise ValueError("paired conversion requires a PA_BBND source case")
    tensors = dict(case_data["tensors"])
    packed = tensors["k"]
    keys = []
    scales = []
    boundaries = tensors["cu_k"].tolist()
    page_size = params["block_size"]
    for batch_index, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        length = end - begin
        page_count = (length + page_size - 1) // page_size
        table = tensors["block_table"]
        pages = (
            torch.arange(page_count)
            if table is None
            else table[batch_index, :page_count].long()
        )
        selected = packed.index_select(0, pages)
        keys.append(selected[..., :512].reshape(-1, 1, PACKED_D)[:length])
        scales.append(selected[..., 512:].reshape(-1, 1, 2, 2)[:length])
    tensors["k"] = torch.cat(keys).contiguous()
    tensors["descale_k"] = torch.cat(scales).contiguous()
    tensors["block_table"] = None
    for name in ("_fused_storage", "_k_storage", "_scale_storage"):
        tensors[name] = None
    params["layout_k"] = "TND"
    params["storage_mode"] = "contiguous"
    return {**case_data, "params": params, "tensors": tensors, "cpu_result": None}


def qsli_golden_rows(case_data, selected_rows):
    params, tensors = case_data["params"], case_data["tensors"]
    return quant_sparse_lightning_indexer_golden_selected(
        tensors["q"],
        tensors["k"],
        tensors["weights"],
        tensors["descale_q"],
        tensors["candidate"],
        tensors["candidate_length"],
        selected_rows,
        descale_k=tensors.get("descale_k"),
        cu_seqlens_q=tensors["cu"],
        cu_seqlens_k=tensors["cu_k"],
        seqused_q=tensors["used_q"],
        seqused_k=tensors["used_k"],
        cmp_residual_k=tensors["residual"],
        block_table=tensors["block_table"],
        output_idx_offset=tensors["offset"],
        topk=params["topk"],
        candidate_block_size=params["candidate_block_size"],
        quant_mode=params["quant_mode"],
        max_seqlen_q=params["max_seqlen_q"],
        mask_mode=params["mask_mode"],
        cmp_ratio=params["cmp_ratio"],
        layout_q=params["layout_q"],
        layout_k=params["layout_k"],
        return_value=params["return_value"],
    )


def _to_npu_tensors(tensors):
    # Legacy PT fixtures retain independent CPU golden inputs. Pack before
    # device execution; the operator itself never launches a packing kernel.
    npu = {}
    ignored = {"k", "descale_k", "_fused_storage", "_k_storage", "_scale_storage"}
    for name, value in tensors.items():
        if name not in ignored:
            npu[name] = value.npu() if isinstance(value, torch.Tensor) else value
    if tensors["k"].ndim == 3 and tensors["k"].shape[-1] == PACKED_D:
        rows = tensors["k"].shape[0]
        if tensors["_k_storage"] is not None:
            storage = tensors["_k_storage"].npu()
            npu["k"] = storage[:, : tensors["k"].reshape(rows, -1).shape[1]].view_as(
                tensors["k"]
            )
        else:
            npu["k"] = tensors["k"].npu()
        if tensors["_scale_storage"] is not None:
            storage = tensors["_scale_storage"].npu()
            npu["descale_k"] = storage[
                :, : tensors["descale_k"].reshape(rows, -1).shape[1]
            ].view_as(tensors["descale_k"])
        else:
            npu["descale_k"] = tensors["descale_k"].npu()
        return npu
    key = tensors["k"]
    if key.ndim == 3:
        packed = key
    else:
        pages, page_size = key.shape[:2]
        packed = torch.cat(
            (
                key.reshape(pages, page_size // BLOCK_SIZE, BLOCK_SIZE * PACKED_D),
                tensors["descale_k"]
                .view(torch.uint8)
                .reshape(pages, page_size // BLOCK_SIZE, BLOCK_SIZE * 4),
            ),
            dim=-1,
        )
    padding = (
        32
        if any(
            tensors.get(name) is not None
            for name in ("_fused_storage", "_k_storage", "_scale_storage")
        )
        else 0
    )
    if padding:
        pages = packed.shape[0]
        width = packed[0].numel()
        storage = torch.empty((pages, width + padding), dtype=torch.uint8)
        storage[:, :width] = packed.reshape(pages, width)
        npu["k"] = storage.npu()[:, :width].view_as(packed)
    else:
        npu["k"] = packed.npu()
    return npu


def run_qsli_case(case_data, device_id=0):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from indexer_test_runtime import configure

    configure()
    import torch_npu
    import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl

    torch_npu.npu.set_device(device_id)
    p, tensors = case_data["params"], case_data["tensors"]
    n = _to_npu_tensors(tensors)
    metadata = (
        torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata(
            n["candidate_length"],
            cu_seqlens_q=n["cu"],
            cu_seqlens_k=n["cu_k"],
            seqused_q=n["used_q"],
            seqused_k=n["used_k"],
            cmp_residual_k=n["residual"],
            num_heads_q=N1,
            num_heads_k=N2,
            head_dim=PACKED_D * 2,
            topk=p["topk"],
            quant_mode=p["quant_mode"],
            candidate_block_size=p["candidate_block_size"],
            max_seqlen_q=p["max_seqlen_q"],
            max_seqlen_k=p["sequence_length"],
            mask_mode=p["mask_mode"],
            cmp_ratio=p["cmp_ratio"],
            layout_q=p["layout_q"],
            layout_k=p["layout_k"],
        )
    )
    actual_outputs = torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer(
        n["q"],
        n["k"],
        n["weights"],
        n["descale_q"],
        n["candidate"],
        n["candidate_length"],
        p["topk"],
        p["quant_mode"],
        p["candidate_block_size"],
        descale_k=n.get("descale_k"),
        cu_seqlens_q=n["cu"],
        cu_seqlens_k=n["cu_k"],
        seqused_q=n["used_q"],
        seqused_k=n["used_k"],
        cmp_residual_k=n["residual"],
        block_table=n["block_table"],
        output_idx_offset=n["offset"],
        metadata=metadata,
        max_seqlen_q=p["max_seqlen_q"],
        mask_mode=p["mask_mode"],
        cmp_ratio=p["cmp_ratio"],
        layout_q=p["layout_q"],
        layout_k=p["layout_k"],
        return_value=p["return_value"],
    )
    torch.npu.synchronize()
    result = {
        name: value.cpu()
        for name, value in zip(("sparse_indices", "sparse_values"), actual_outputs)
    }
    selected_rows = tensors.get("_selected_rows")
    if selected_rows is not None:
        selected = torch.tensor(selected_rows, dtype=torch.long)
        result = {
            name: value.index_select(0, selected) if value.ndim == 3 else value
            for name, value in result.items()
        }
    return result


def _remove_offset(indices, offset):
    result = indices.clone()
    if offset is None:
        return result
    expanded = offset.reshape(indices.shape[0], 1, 1).expand_as(indices)
    valid = result >= 0
    result[valid] -= expanded[valid]
    return result


def compare_qsli_case(case_data, actual):
    p, golden = case_data["params"], case_data["cpu_result"]
    offset = golden["_output_idx_offset"]
    expected_idx = _remove_offset(golden["sparse_indices"], offset)
    actual_idx = _remove_offset(actual["sparse_indices"], offset)
    rows, s2 = actual_idx.shape[0], p["sequence_length"]
    selected_rows = case_data["tensors"].get("_selected_rows")
    batch = rows if selected_rows is not None else p["batch_size"]
    cu_tensor = case_data["tensors"]["cu"]
    used_k_tensor = case_data["tensors"]["used_k"]
    # The operator accepts both tensors as optional inputs.  The golden path
    # materializes the implicit defaults, while the NPU call must still receive
    # the original None values so the optional-input profiles are exercised.
    cu = (
        golden["_cu_seqlens_q"].tolist()
        if selected_rows is not None
        else (
            golden["_cu_seqlens_q"].tolist()
            if cu_tensor is None
            else cu_tensor.tolist()
        )
    )
    used_k = (
        golden["_seqused_k"].tolist()
        if selected_rows is not None
        else ([s2] * batch if used_k_tensor is None else used_k_tensor.tolist())
    )
    page_size = int(p["block_size"])
    compare_params = (
        batch,
        1 if selected_rows is not None else p["query_length"],
        s2,
        rows,
        batch * s2,
        N1,
        N2,
        128,
        page_size,
        (s2 + page_size - 1) // page_size,
        torch.uint8,
        torch.float32,
        torch.uint8,
        torch.int32,
        cu,
        used_k,
        1,
        1,
        "TND",
        "PA_BBND",
        p["topk"],
        p["mask_mode"],
        (-1, 1),
        (-1, 1),
        (0, 1),
        (125, 129),
        (125, 129),
    )
    result, fulfill = result_compare_method.check_result(
        expected_idx,
        actual_idx,
        golden["_topk_value"].numpy(),
        compare_params,
        return_value=p["return_value"],
        cpu_topk_value=golden["sparse_values"] if p["return_value"] else None,
        npu_topk_value=actual["sparse_values"] if p["return_value"] else None,
    )
    if result != "Pass":
        raise AssertionError(f"index result={result}, fulfill_percent={fulfill}")
    if not p["return_value"]:
        if tuple(actual["sparse_values"].shape) != (0,):
            raise AssertionError(
                "return_value=False must produce an empty sparse_values tensor"
            )
        return "Pass", fulfill
    if actual["sparse_values"].shape != actual["sparse_indices"].shape:
        raise AssertionError(
            "return_value=True requires sparse_values to match sparse_indices shape"
        )
    reference = torch.zeros_like(actual["sparse_values"], dtype=torch.float32)
    golden_cu = golden["_cu_seqlens_q"].tolist()
    for batch, (begin, end) in enumerate(zip(golden_cu, golden_cu[1:])):
        for row in range(begin, end):
            valid = actual_idx[row, 0] >= 0
            reference[row, 0, valid] = golden["_topk_value"][
                batch, 0, row - begin, actual_idx[row, 0, valid].long()
            ]
    value_pass = result_compare_method.judge_value_by_isclose(
        actual["sparse_values"].float().numpy().reshape(-1),
        reference.numpy().reshape(-1),
        force_bf16=True,
    )
    if not value_pass:
        raise AssertionError("sparse_values failed result_compare_method")
    return "Pass", fulfill


def compare_qsli_all_rows(case_data, actual, *, peer_case=None, peer_actual=None):
    total_rows = case_data["tensors"]["q"].shape[0]
    fulfill_min = 100.0
    for begin in range(0, total_rows, 32):
        rows = list(range(begin, min(begin + 32, total_rows)))
        golden = qsli_golden_rows(case_data, rows)
        chunk_case = {
            **case_data,
            "cpu_result": golden,
            "tensors": {**case_data["tensors"], "_selected_rows": rows},
        }
        chunk_actual = {
            name: value[begin : begin + len(rows)] if value.ndim == 3 else value
            for name, value in actual.items()
        }
        _, fulfill = compare_qsli_case(chunk_case, chunk_actual)
        fulfill_min = min(fulfill_min, fulfill)
        if peer_case is not None:
            peer_golden = qsli_golden_rows(peer_case, rows)
            for name in ("_topk_value", "_seqused_k", "_output_idx_offset"):
                if not torch.equal(golden[name], peer_golden[name]):
                    raise AssertionError(
                        f"PA/TND logical golden mismatch: {name}, rows {begin}:{begin + len(rows)}"
                    )
            peer_chunk = {
                **peer_case,
                "cpu_result": peer_golden,
                "tensors": {**peer_case["tensors"], "_selected_rows": rows},
            }
            peer_values = {
                name: value[begin : begin + len(rows)] if value.ndim == 3 else value
                for name, value in peer_actual.items()
            }
            _, fulfill = compare_qsli_case(peer_chunk, peer_values)
            fulfill_min = min(fulfill_min, fulfill)
    return "Pass", fulfill_min


class QsliCaseSelector:
    @staticmethod
    def natural_key(path):
        return [
            int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", Path(path).name)
        ]

    @staticmethod
    def parse_indexes(expression, total):
        indexes = []
        for token in str(expression or "").split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"invalid descending case index range: {token}")
                indexes.extend(range(start, end + 1))
            else:
                indexes.append(int(token))
        invalid = [index for index in indexes if index < 1 or index > total]
        if invalid:
            raise ValueError(f"case indexes out of range 1..{total}: {invalid}")
        return indexes

    @classmethod
    def resolve(cls, pt_dir, explicit_files="", case_names="", case_indexes=""):
        files = (
            [Path(x) for x in explicit_files.split(",") if x]
            if explicit_files
            else sorted(Path(pt_dir).glob("*.pt"), key=cls.natural_key)
        )
        if case_names:
            wanted = [x.strip() for x in case_names.split(",") if x.strip()]
            by_name = {x.stem: x for x in files}
            files = [by_name[x] for x in wanted]
        if case_indexes:
            indexes = cls.parse_indexes(case_indexes, len(files))
            files = [files[x - 1] for x in indexes]
        if not files:
            raise ValueError(f"no PT cases found in: {pt_dir}")
        return files
