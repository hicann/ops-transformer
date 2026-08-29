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

import logging
import math
import os
import random

import torch

import check_valid_param
import combined_kv_cache


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DATA_RANGE_LEFT = -1000
DATA_RANGE_RIGHT = 1000
FP8_E4M3_MAX = 448.0
SCALE_EPSILON = 1e-8
EMPTY_LSE = -3.4028234663852886e38
SOFTMAX_MAX_SENTINEL = EMPTY_LSE
MASK_VALUE = -10000.0


def _fp8_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for quant_block_sparse_attn FP8 test data"
        )
    return torch.float8_e4m3fn


def _torch_dtype(value):
    if value in (torch.bfloat16, "torch.bfloat16", "bfloat16", "bf16"):
        return torch.bfloat16
    return value


def _fp8_normalize_data_range(data_range):
    """Normalize a TTK range while retaining pytest's None/default semantics."""
    if data_range is None:
        return None
    if isinstance(data_range, (list, tuple)):
        if len(data_range) == 0:
            return None
        if len(data_range) != 2:
            raise ValueError(
                f"data_range must be empty, a scalar, or [min, max], got {data_range}"
            )
        low, high = data_range
        if low is None and high is None:
            return None
        return (float(low), float(high))
    return float(data_range)


def _fp8_assemble_case(
    query,
    key,
    sparse_indices,
    block_table,
    *,
    softmax_scale,
    sparse_q_block_size,
    sparse_kv_block_size,
    layout_q,
    layout_kv,
    layout_sparse_indices,
    layout_out,
    mask_mode,
    return_softmax_lse,
    sparse_mode,
    cu_seqlens_q_value,
    cu_seqlens_kv_value,
    seqused_q_value,
    seqused_kv_value,
    p_scale_value,
    seed,
    blocksize,
    data_range_q,
    data_range_k,
    data_range_v,
    testcase_name,
    sparse_pattern="sequential",
    block_table_pattern="sequential",
    **kwargs,
):
    """Translate the shared 16-tensor TTK case into the pytest FP8 schema."""
    del kwargs
    cu_q = [int(value) for value in cu_seqlens_q_value]
    cu_kv = [int(value) for value in cu_seqlens_kv_value]
    seq_q = [int(value) for value in seqused_q_value]
    seq_kv = [int(value) for value in seqused_kv_value]
    if len(cu_q) < 2 or cu_q[0] != 0:
        raise ValueError("cu_seqlens_q_value must start with 0 and contain B+1 values")

    q_lengths = [end - start for start, end in zip(cu_q, cu_q[1:])]
    batch = int(sparse_indices.shape[0])
    if len(q_lengths) != batch or len(seq_kv) != batch:
        raise ValueError(
            "cu_seqlens_q_value and seqused_kv_value must match the sparse batch size"
        )
    if cu_kv:
        raise ValueError("cu_seqlens_kv_value must be empty for FP8 TTK cases")
    if seq_q:
        raise ValueError("seqused_q_value must be empty for FP8 TTK cases")

    if layout_q == "NTD":
        num_heads_q = int(query.shape[0])
    elif layout_q == "TND":
        num_heads_q = int(query.shape[1])
    else:
        raise ValueError(f"unsupported FP8 layout_q: {layout_q!r}")

    return {
        "Testcase_Name": testcase_name,
        "B": batch,
        "S1": max(q_lengths),
        "S2": max(seq_kv),
        "N1": num_heads_q,
        "N2": int(key.shape[1]),
        "D": int(query.shape[-1]),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_sparse_indices": layout_sparse_indices,
        "layout_out": layout_out,
        "output_dtype": torch.bfloat16,
        "quant_mode": 1,
        "mask_mode": int(mask_mode),
        "return_softmax_lse": bool(return_softmax_lse),
        "softmax_scale": float(softmax_scale),
        "sparse_q_block_size": int(sparse_q_block_size),
        "sparse_kv_block_size": int(sparse_kv_block_size),
        "sparse_mode": sparse_mode,
        "sparse_pattern": sparse_pattern,
        "block_table_pattern": block_table_pattern,
        "block_size": int(blocksize) if int(blocksize) > 0 else int(key.shape[2]),
        "block_num": int(key.shape[0]),
        "max_block_per_batch": int(block_table.shape[1]),
        "cu_seqlens_q_value": cu_q,
        "cu_seqlens_kv_value": cu_kv,
        "seqused_q_value": seq_q,
        "seqused_kv_value": seq_kv,
        "q_datarange": _fp8_normalize_data_range(data_range_q),
        "k_datarange": _fp8_normalize_data_range(data_range_k),
        "v_datarange": _fp8_normalize_data_range(data_range_v),
        "p_scale_value": (None if p_scale_value is None else float(p_scale_value)),
        "pa_block_padding_bytes": 0,
        "seed": int(seed),
        "S1EQS2": False,
    }


def _normalize_params(params):
    if not isinstance(params, dict):
        raise ValueError("quant_block_sparse_attn params should be a dict")

    normalized = dict(params)
    normalized.setdefault("Testcase_Name", None)
    normalized.setdefault("layout_q", "TND")
    normalized.setdefault("layout_kv", "PA_BNBD")
    normalized.setdefault("output_dtype", torch.bfloat16)
    normalized.setdefault("sparse_q_block_size", 128)
    normalized.setdefault("sparse_kv_block_size", 128)
    normalized.setdefault("layout_sparse_indices", "B_N_Qb_Kb")
    normalized.setdefault("layout_out", "TND")
    normalized.setdefault("quant_mode", 1)
    normalized.setdefault("mask_mode", 3)
    normalized.setdefault("return_softmax_lse", False)
    normalized.setdefault("S1EQS2", False)
    normalized.setdefault("seed", 0)
    if normalized.get("sparse_mode") is None:
        normalized["sparse_mode"] = "random"
    normalized.setdefault("sparse_pattern", "sequential")
    normalized.setdefault("block_table_pattern", "sequential")
    normalized.setdefault("block_num", None)
    normalized.setdefault("q_datarange", None)
    normalized.setdefault("k_datarange", None)
    normalized.setdefault("v_datarange", None)
    normalized.setdefault("pa_block_padding_bytes", 0)
    normalized.setdefault("p_scale_value", 1.0)
    normalized["output_dtype"] = _torch_dtype(normalized["output_dtype"])
    for key in (
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "sparse_q_block_size",
        "sparse_kv_block_size",
        "seed",
        "block_num",
        "max_block_per_batch",
        "pa_block_padding_bytes",
    ):
        if isinstance(normalized.get(key), float) and normalized[key].is_integer():
            normalized[key] = int(normalized[key])
    sparse_mode = normalized["sparse_mode"]
    if isinstance(sparse_mode, str):
        sparse_mode = sparse_mode.strip().lower()
        normalized["sparse_mode"] = sparse_mode
    normalized["softmax_scale"] = normalized.get("softmax_scale") or (
        1.0 / math.sqrt(normalized["D"])
    )
    return normalized


def _make_lengths(case):
    batch = case["B"]
    cu_seqlens_q_value = case.get("cu_seqlens_q_value")
    seqused_kv_value = case.get("seqused_kv_value")
    if not isinstance(cu_seqlens_q_value, list):
        raise ValueError("cu_seqlens_q_value should be an integer list")
    if not isinstance(seqused_kv_value, list):
        raise ValueError("seqused_kv_value should be an integer list")
    if len(cu_seqlens_q_value) != batch + 1:
        raise ValueError(
            f"cu_seqlens_q_value length should be B + 1, got {len(cu_seqlens_q_value)}"
        )
    if len(seqused_kv_value) != batch:
        raise ValueError(
            f"seqused_kv_value length should be B, got {len(seqused_kv_value)}"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in cu_seqlens_q_value + seqused_kv_value
    ):
        raise ValueError("sequence length values should all be integers")
    if cu_seqlens_q_value[0] != 0:
        raise ValueError("cu_seqlens_q_value should start at 0")

    q_lengths = [
        cu_seqlens_q_value[index + 1] - cu_seqlens_q_value[index]
        for index in range(batch)
    ]
    if any(length < 0 or length > case["S1"] for length in q_lengths):
        raise ValueError(f"query lengths should be in [0, S1], got {q_lengths}")
    kv_lengths = list(seqused_kv_value)
    if any(length < 0 or length > case["S2"] for length in kv_lengths):
        raise ValueError(f"KV lengths should be in [0, S2], got {kv_lengths}")
    return q_lengths, max(q_lengths), kv_lengths, max(kv_lengths)


def _source_absmax():
    return max(max(abs(DATA_RANGE_LEFT), abs(DATA_RANGE_RIGHT)), 1.0)


def _parse_value_range(value):
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        try:
            r = float(s)
            return -r, r
        except ValueError:
            pass
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"datarange should be like [-100,100], got {value!r}")
        return float(parts[0].strip()), float(parts[1].strip())
    if isinstance(value, (int, float)):
        r = float(value)
        return -r, r
    return float(value[0]), float(value[1])


def _rand_value_range(shape, low, high, generator):
    return torch.empty(shape, dtype=torch.float32).uniform_(
        low, high, generator=generator
    )


def _make_source(
    case, key, shape, default_low, default_high, generator, amp_shape=None
):
    value = case.get(key)
    if value is None:
        source = _rand_fullquant_source(
            shape, default_low, default_high, generator, amp_shape=amp_shape
        )
        return source, FP8_E4M3_MAX
    low, high = _parse_value_range(value)
    if low > high:
        raise ValueError(
            f"{key} datarange should satisfy low <= high, got ({low}, {high})"
        )
    if not math.isfinite(low) or not math.isfinite(high):
        if low == high or (math.isnan(low) and math.isnan(high)):
            return torch.full(shape, low, dtype=torch.float32), FP8_E4M3_MAX
        finite_low = min(high, 0.0) - 1.0 if math.isinf(low) else low
        finite_high = max(low, 0.0) + 1.0 if math.isinf(high) else high
        source = _rand_value_range(shape, finite_low, finite_high, generator)
        flattened = source.reshape(-1)
        flattened[0], flattened[-1] = low, high
        return source, FP8_E4M3_MAX
    source = _rand_value_range(shape, low, high, generator)
    return source, max(abs(low), abs(high))


def _log_uniform_amplitude(shape, low, high, generator):
    high = max(float(high), SCALE_EPSILON)
    low = max(min(float(low), high), SCALE_EPSILON)
    log_low = math.log10(low)
    log_high = math.log10(high)
    exponent = torch.empty(shape, dtype=torch.float32).uniform_(
        log_low, log_high, generator=generator
    )
    return torch.pow(torch.tensor(10.0, dtype=torch.float32), exponent)


def _rand_fullquant_source(shape, amp_low, amp_high, generator, amp_shape=None):
    base = torch.empty(shape, dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )
    if amp_shape is None:
        amp_shape = shape[:-1] + (1,)
    amps = _log_uniform_amplitude(amp_shape, amp_low, amp_high, generator)
    return base * amps


def _quant_fp32_to_fp8(tensor, quant_scale):
    quantized = tensor.to(torch.float32) * quant_scale
    quantized = torch.where(torch.isposinf(tensor), FP8_E4M3_MAX, quantized)
    quantized = torch.where(torch.isneginf(tensor), -FP8_E4M3_MAX, quantized)
    quantized = torch.clamp(quantized, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quantized.to(_fp8_dtype()).contiguous()


def _quantize_per_token_head(tensor, max_abs=FP8_E4M3_MAX):
    max_abs = min(float(max_abs), FP8_E4M3_MAX)
    finite_abs = torch.where(torch.isfinite(tensor), torch.abs(tensor), 0.0)
    row_max = finite_abs.amax(dim=-1, keepdim=True)
    row_max = torch.maximum(
        row_max, torch.tensor(SCALE_EPSILON, dtype=torch.float32, device=tensor.device)
    )
    quant_scale = max_abs / row_max
    descale = 1.0 / quant_scale
    descale = torch.where(
        torch.isinf(tensor).any(dim=-1, keepdim=True), float("inf"), descale
    )
    return _quant_fp32_to_fp8(tensor, quant_scale), descale.squeeze(-1).contiguous()


def _quantize_value_per_head(tensor, max_abs=FP8_E4M3_MAX):
    max_abs = min(float(max_abs), FP8_E4M3_MAX)
    finite_abs = torch.where(torch.isfinite(tensor), torch.abs(tensor), 0.0)
    head_max = finite_abs.amax(dim=(0, 1, 3), keepdim=True)
    head_max = torch.maximum(
        head_max, torch.tensor(SCALE_EPSILON, dtype=torch.float32, device=tensor.device)
    )
    quant_scale = max_abs / head_max
    value = _quant_fp32_to_fp8(tensor, quant_scale)
    descale = 1.0 / quant_scale
    has_inf = torch.isinf(tensor).any(dim=(0, 1, 3), keepdim=True)
    descale = torch.where(has_inf, float("inf"), descale)
    return value, descale.reshape(tensor.shape[2]).contiguous()


def _finalize_fp8_attention_rows(
    acc,
    softmax_sum,
    softmax_max,
    softmax_max_offset,
    v_scale,
    single_chunk,
):
    """Match the kernel's final zero-sum and invalid-row guards."""
    active = (softmax_max != SOFTMAX_MAX_SENTINEL) & (softmax_sum != 0)
    safe_sum = torch.where(active, softmax_sum, torch.ones_like(softmax_sum))
    numerator = acc * v_scale if single_chunk else acc
    attention_out = numerator / safe_sum.view(-1, 1)
    attention_out = torch.where(
        active.view(-1, 1), attention_out, torch.zeros_like(attention_out)
    )
    softmax_lse = torch.log(safe_sum) + softmax_max_offset
    softmax_lse = torch.where(
        active,
        softmax_lse,
        torch.full_like(softmax_lse, SOFTMAX_MAX_SENTINEL),
    )
    return attention_out, softmax_lse


def _make_block_table(
    batch,
    seqused_kv,
    block_size,
    pattern,
    rng,
    blocknum=None,
    max_block_per_batch=None,
):
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError(f"batch should be a positive int, got {batch}")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size should be a positive int, got {block_size}")
    if not isinstance(max_block_per_batch, int) or max_block_per_batch <= 0:
        raise ValueError(
            f"max_block_per_batch should be a positive int, got {max_block_per_batch}"
        )
    if not isinstance(blocknum, int) or blocknum <= 0:
        raise ValueError(f"blocknum should be a positive int, got {blocknum}")

    if pattern not in ("random", "sequential"):
        raise ValueError(f"unsupported block_table_pattern: {pattern}")

    total_slots = batch * max_block_per_batch
    if pattern == "random":
        physical_ids = [rng.randrange(blocknum) for _ in range(total_slots)]
    else:
        physical_ids = [index % blocknum for index in range(total_slots)]

    return torch.tensor(physical_ids, dtype=torch.int32).reshape(
        batch, max_block_per_batch
    )


def _allowed_blocks(
    mask_mode, qb_idx, sparse_q_block_size, sparse_kv_block_size, q_len, kv_len
):
    block_num = math.ceil(kv_len / sparse_kv_block_size)
    if block_num <= 0:
        return []
    if mask_mode == 0:
        return list(range(block_num))
    if mask_mode != 3:
        raise ValueError(f"unsupported mask_mode: {mask_mode}")

    max_token = (qb_idx + 1) * sparse_q_block_size - 1
    max_token += kv_len - q_len
    if max_token < 0:
        return []
    max_block = min(block_num - 1, max_token // sparse_kv_block_size)
    return list(range(max_block + 1))


def _select_blocks(blocks, sparse_count, pattern, rng):
    if sparse_count <= 0 or not blocks or pattern == "empty":
        return []
    if pattern in ("sequential", "dense", "causal"):
        return blocks[: min(sparse_count, len(blocks))]
    if pattern == "reverse":
        return list(reversed(blocks[-min(sparse_count, len(blocks)) :]))
    if pattern == "tail":
        selected = blocks[: max(0, min(sparse_count, len(blocks)) - 1)]
        if blocks[-1] not in selected:
            selected.append(blocks[-1])
        return selected[:sparse_count]
    if pattern == "random":
        selected = blocks[:]
        rng.shuffle(selected)
        return selected[: min(sparse_count, len(selected))]
    if pattern == "empty_tail":
        return blocks[: min(sparse_count, len(blocks))]
    raise ValueError(f"unsupported sparse_pattern: {pattern}")


def _make_sparse_indices(case, q_lengths, kv_lengths, rng):
    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    group = n1 // n2
    qb_max = math.ceil(case["S1"] / case["sparse_q_block_size"])
    kv_max = math.ceil(case["S2"] / case["sparse_kv_block_size"])
    sparse_mode = case["sparse_mode"]
    if sparse_mode not in ("dense", "random"):
        raise ValueError(f"unsupported sparse_mode: {sparse_mode}")
    sparse_counts = []
    for batch_idx in range(batch):
        batch_kv_max = math.ceil(kv_lengths[batch_idx] / case["sparse_kv_block_size"])
        sparse_count = (
            batch_kv_max if sparse_mode == "dense" else rng.randint(0, batch_kv_max)
        )
        sparse_counts.append(sparse_count)
    sparse_indices = torch.full(
        (batch, n1, qb_max, kv_max), fill_value=-1, dtype=torch.int32
    )
    sparse_seq_len = torch.zeros((batch, n1, qb_max), dtype=torch.int32)

    for batch_idx in range(batch):
        sparse_count = sparse_counts[batch_idx]
        qb_batch = math.ceil(q_lengths[batch_idx] / case["sparse_q_block_size"])
        for qb_idx in range(qb_max):
            allowed = _allowed_blocks(
                case["mask_mode"],
                qb_idx,
                case["sparse_q_block_size"],
                case["sparse_kv_block_size"],
                q_lengths[batch_idx],
                kv_lengths[batch_idx],
            )
            for n2_idx in range(n2):
                for group_idx in range(group):
                    head_idx = n2_idx * group + group_idx
                    if (
                        case["sparse_pattern"] == "empty_tail"
                        and qb_idx == qb_batch - 1
                    ) or qb_idx >= qb_batch:
                        selected = []
                    else:
                        selected = _select_blocks(
                            allowed, sparse_count, case["sparse_pattern"], rng
                        )
                    sparse_seq_len[batch_idx, head_idx, qb_idx] = len(selected)
                    if selected:
                        sparse_indices[batch_idx, head_idx, qb_idx, : len(selected)] = (
                            torch.tensor(selected, dtype=torch.int32)
                        )
    return sparse_indices, sparse_seq_len


def _query_index(cu_seqlens_q, batch_idx, q_idx):
    return int(cu_seqlens_q[batch_idx].item()) + q_idx


def _set_output(output, cu_seqlens_q, batch_idx, q_idx, head_idx, value):
    output[_query_index(cu_seqlens_q, batch_idx, q_idx), head_idx] = value


def _set_lse(softmax_lse, cu_seqlens_q, batch_idx, q_idx, head_idx, value):
    token_idx = _query_index(cu_seqlens_q, batch_idx, q_idx)
    softmax_lse[head_idx, token_idx] = value


def _make_fullquant_tensors(case, cu_seqlens_q, seqused_kv, generator):
    layout_q = case["layout_q"]
    total_q = int(cu_seqlens_q[-1].item())
    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    head_dim = case["D"]
    amp_high = _source_absmax()

    if layout_q == "NTD":
        query_shape = (n1, total_q, head_dim)
    else:
        query_shape = (total_q, n1, head_dim)
    query_source, q_max = _make_source(
        case, "q_datarange", query_shape, amp_high * 0.01, amp_high, generator
    )
    query, q_scale = _quantize_per_token_head(query_source, q_max)

    # KV 实际长度以 seqused_kv 为准（与 kernel 读取一致），不再依赖 cu_seqlens_kv 构造扁平化 k_scale
    kv_shape = (batch, case["S2"], n2, head_dim)
    dense_key_source, k_max = _make_source(
        case, "k_datarange", kv_shape, 1.0, amp_high, generator
    )
    dense_key, dense_k_scale_base = _quantize_per_token_head(dense_key_source, k_max)
    dense_k_scale = torch.zeros((batch, case["S2"], n2), dtype=torch.float32)
    for batch_idx in range(batch):
        kv_len = int(seqused_kv[batch_idx].item())
        dense_k_scale[batch_idx, :kv_len] = dense_k_scale_base[batch_idx, :kv_len]

    dense_value_source, v_max = _make_source(
        case,
        "v_datarange",
        kv_shape,
        1.0,
        amp_high,
        generator,
        amp_shape=(batch, 1, n2, 1),
    )
    dense_value, v_scale = _quantize_value_per_head(dense_value_source, v_max)
    v_scale = v_scale.contiguous()

    p_scale = (
        None
        if case["p_scale_value"] is None
        else torch.tensor([float(case["p_scale_value"])], dtype=torch.float32)
    )
    return query, dense_key, dense_value, q_scale, dense_k_scale, v_scale, p_scale


def _mask_positions(case, q_idx, positions, q_len, kv_len):
    if case["mask_mode"] == 0:
        return torch.ones((len(positions),), dtype=torch.bool)
    if case["mask_mode"] != 3:
        raise ValueError(f"unsupported mask_mode: {case['mask_mode']}")
    limit = q_idx + kv_len - q_len
    return torch.tensor([pos <= limit for pos in positions], dtype=torch.bool)


def _positions_from_sparse(
    case, sparse_indices, sparse_seq_len, batch_idx, head_idx, qb_idx, kv_len
):
    block_count = int(sparse_seq_len[batch_idx, head_idx, qb_idx].item())
    chunk_positions = []
    i = 0
    while i < block_count:
        pair_positions = []
        for j in range(2):
            if i + j < block_count:
                block_idx = int(
                    sparse_indices[batch_idx, head_idx, qb_idx, i + j].item()
                )
                if block_idx >= 0:
                    start = block_idx * case["sparse_kv_block_size"]
                    end = min(start + case["sparse_kv_block_size"], kv_len)
                    if start < kv_len:
                        pair_positions.extend(range(start, end))
        if pair_positions:
            chunk_positions.append(pair_positions)
        i += 2
    return chunk_positions


def _count_mask_valid_pairs(case, q_start, q_end, kv_start, kv_end, q_len, kv_len):
    q_count = max(0, q_end - q_start)
    kv_count = max(0, kv_end - kv_start)
    if q_count == 0 or kv_count == 0:
        return 0
    if case["mask_mode"] == 0:
        return q_count * kv_count
    if case["mask_mode"] != 3:
        raise ValueError(f"unsupported mask_mode: {case['mask_mode']}")

    valid_pairs = 0
    causal_offset = kv_len - q_len
    for q_idx in range(q_start, q_end):
        valid_kv_end = min(kv_end, q_idx + causal_offset + 1)
        if valid_kv_end > kv_start:
            valid_pairs += valid_kv_end - kv_start
    return valid_pairs


def _calc_cube_compute_amount(
    case, q_lengths, kv_lengths, sparse_indices, sparse_seq_len
):
    q_block_size = int(case["sparse_q_block_size"])
    kv_block_size = int(case["sparse_kv_block_size"])
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError(
            f"sparse block size must be positive, got q={q_block_size}, kv={kv_block_size}"
        )

    sparse_indices_cpu = torch.as_tensor(sparse_indices).cpu()
    sparse_seq_len_cpu = torch.as_tensor(sparse_seq_len).cpu()
    if sparse_indices_cpu.dim() != 4 or sparse_seq_len_cpu.dim() != 3:
        raise ValueError(
            f"invalid sparse shapes: sparse_indices={tuple(sparse_indices_cpu.shape)}, "
            f"sparse_seq_len={tuple(sparse_seq_len_cpu.shape)}"
        )

    batch = len(q_lengths)
    n1 = int(case["N1"])
    head_dim = int(case["D"])
    if len(kv_lengths) != batch:
        raise ValueError(
            f"q_lengths and kv_lengths batch mismatch: {len(q_lengths)} vs {len(kv_lengths)}"
        )
    if sparse_seq_len_cpu.shape[0] < batch or sparse_indices_cpu.shape[0] < batch:
        raise ValueError(
            f"sparse batch dimension is smaller than seqused batch: sparse_indices={tuple(sparse_indices_cpu.shape)}, "
            f"sparse_seq_len={tuple(sparse_seq_len_cpu.shape)}, batch={batch}"
        )
    if sparse_seq_len_cpu.shape[1] < n1 or sparse_indices_cpu.shape[1] < n1:
        raise ValueError(
            f"sparse head dimension is smaller than N1={n1}: sparse_indices={tuple(sparse_indices_cpu.shape)}, "
            f"sparse_seq_len={tuple(sparse_seq_len_cpu.shape)}"
        )

    basic_block_count = 0
    qb_limit = sparse_seq_len_cpu.shape[2]
    kb_limit = sparse_indices_cpu.shape[3]
    for batch_idx in range(batch):
        q_len = int(q_lengths[batch_idx])
        kv_len = int(kv_lengths[batch_idx])
        qb_count = min(qb_limit, math.ceil(q_len / q_block_size))
        for head_idx in range(n1):
            for qb_idx in range(qb_count):
                q_start = qb_idx * q_block_size
                q_end = min(q_start + q_block_size, q_len)
                if q_start >= q_end:
                    continue
                block_count = int(
                    sparse_seq_len_cpu[batch_idx, head_idx, qb_idx].item()
                )
                block_count = max(0, min(block_count, kb_limit))
                for sparse_idx in range(block_count):
                    kv_block_idx = int(
                        sparse_indices_cpu[
                            batch_idx, head_idx, qb_idx, sparse_idx
                        ].item()
                    )
                    if kv_block_idx < 0:
                        continue
                    kv_start = kv_block_idx * kv_block_size
                    kv_end = min(kv_start + kv_block_size, kv_len)
                    if kv_start >= kv_end:
                        continue
                    basic_block_count += 1

    single_basic_block_compute = q_block_size * kv_block_size * head_dim
    multiply_add_compute = 2
    bmm_compute_count = 2
    cube_compute_amount = (
        basic_block_count
        * single_basic_block_compute
        * multiply_add_compute
        * bmm_compute_count
    )
    return {
        "basic_block_count": basic_block_count,
        "basic_block_shape": (q_block_size, kv_block_size, head_dim),
        "single_basic_block_compute": single_basic_block_compute,
        "multiply_add_compute": multiply_add_compute,
        "bmm_compute_count": bmm_compute_count,
        "cube_compute_amount": cube_compute_amount,
    }


def _calc_cube_compute_capacity():
    unit_conversion = 1000
    fractal_m = 16
    fractal_n = 16
    fp8_fractal_k = 32
    min_fractal_compute = fractal_m * fractal_n * fp8_fractal_k
    frequency_ghz = 1.65
    aic_count = 32
    multiply_add_compute = 2
    cube_compute_capacity = (
        unit_conversion
        * min_fractal_compute
        * frequency_ghz
        * aic_count
        * multiply_add_compute
    )
    return {
        "unit_conversion": unit_conversion,
        "fractal_shape": (fractal_m, fractal_n, fp8_fractal_k),
        "min_fractal_compute": min_fractal_compute,
        "frequency_ghz": frequency_ghz,
        "aic_count": aic_count,
        "multiply_add_compute": multiply_add_compute,
        "cube_compute_capacity": cube_compute_capacity,
    }


def _log_cube_compute_amount(
    case_name, case, q_lengths, kv_lengths, sparse_indices, sparse_seq_len
):
    compute_info = _calc_cube_compute_amount(
        case, q_lengths, kv_lengths, sparse_indices, sparse_seq_len
    )
    capacity_info = _calc_cube_compute_capacity()
    basic_block_shape = compute_info["basic_block_shape"]
    fractal_shape = capacity_info["fractal_shape"]
    mfu_time = (
        compute_info["cube_compute_amount"] / capacity_info["cube_compute_capacity"]
    )
    logger.info("case_name=%s", case_name)
    logger.info("FLOPS计算过程量:")
    logger.info("基本块数量: %d", compute_info["basic_block_count"])
    logger.info(
        "基本块的shape: q_block_size=%d, kv_block_size=%d, head_dim=%d",
        basic_block_shape[0],
        basic_block_shape[1],
        basic_block_shape[2],
    )
    logger.info("单基本块计算量: %d", compute_info["single_basic_block_compute"])
    logger.info(
        "FLOPS计算公式: 基本块数(%d) * 单基本块计算量(%d) * 乘加计算(%d) * 两次bmm计算(%d) = %d",
        compute_info["basic_block_count"],
        compute_info["single_basic_block_compute"],
        compute_info["multiply_add_compute"],
        compute_info["bmm_compute_count"],
        compute_info["cube_compute_amount"],
    )
    logger.info("算力计算过程量:")
    logger.info(
        "一轮cycle对应最小分型shape: m=%d, n=%d, k=%d(fp8为32)",
        fractal_shape[0],
        fractal_shape[1],
        fractal_shape[2],
    )
    logger.info("一轮cycle对应最小分型计算量: %d", capacity_info["min_fractal_compute"])
    logger.info(
        "算力计算公式: 单位换算(%d) * (一轮cycle对应最小分型计算量(%d) * 频率GHz(%.2f) * "
        "AIC数量(%d) * 乘加计算(%d)) = %.6f",
        capacity_info["unit_conversion"],
        capacity_info["min_fractal_compute"],
        capacity_info["frequency_ghz"],
        capacity_info["aic_count"],
        capacity_info["multiply_add_compute"],
        capacity_info["cube_compute_capacity"],
    )
    logger.info(
        "MFU*时间计算公式: FLOPS(%d) / 算力(%.6f) = MFU * 时间(us) = %.6f",
        compute_info["cube_compute_amount"],
        capacity_info["cube_compute_capacity"],
        mfu_time,
    )
    return compute_info["cube_compute_amount"], mfu_time


def _reference_attention(
    case,
    query,
    kv_cache_storage,
    kv_cache_meta,
    block_table,
    q_scale,
    v_scale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    cu_seqlens_q,
    q_lengths,
    kv_lengths,
):
    # 向量化版本（采纳 MR!40 的 einops 思路把 query token 维度批量进 matmul），
    # 但严格保留参考实现的数据流以保证逐位一致：
    #  - 标度在 QK matmul 之后乘到 score 上（q_scale*k_scale*softmax_scale），不在 matmul 前缩放 q/k；
    #  - 沿 KV 轴按「2 个 sparse block」(s2LoopCount*2) 分块的 online/flash FP8 量化（运行中局部 max）；
    #  - 按 sparse_indices 中 block 的出现顺序拼接 positions（random/reverse/tail 非物理升序）来分块；
    #  - 实际长度掩码（q>=q_len、kv>=kv_len 无效；causal 对角线为 kv_len - q_len）；
    #  - 全掩码空行：输出 0、lse = EMPTY_LSE。
    query = query.to(torch.float32)
    kv_key, kv_value, kv_k_scale = combined_kv_cache.make_combined_kv_views(
        kv_cache_storage, kv_cache_meta
    )
    # fp8 视图不支持张量索引（"index_cpu" not implemented for float8_e4m3），
    # 先在循环外整体转成 float32（fp8->fp32 无损）再按索引取值
    kv_key = kv_key.to(torch.float32)
    kv_value = kv_value.to(torch.float32)

    layout_q = case["layout_q"]
    batch = case["B"]
    n1 = case["N1"]
    n2 = case["N2"]
    group = n1 // n2
    head_dim = case["D"]
    output_dtype = case["output_dtype"]
    fp8_dtype = _fp8_dtype()
    softmax_scale = float(case["softmax_scale"])
    p_scale_value = 1.0 if p_scale is None else float(p_scale[0].item())
    ln_p_scale = torch.log(torch.tensor(p_scale_value, dtype=torch.float32)).item()
    sparse_q_block_size = case["sparse_q_block_size"]
    sparse_kv_block_size = case["sparse_kv_block_size"]
    total_q = int(cu_seqlens_q[-1].item())
    attention_out = torch.zeros((total_q, n1, head_dim), dtype=output_dtype)
    softmax_lse = torch.full((n1, total_q), EMPTY_LSE, dtype=torch.float32)

    qb_max = math.ceil(case["S1"] / sparse_q_block_size)
    for batch_idx in range(batch):
        q_len = q_lengths[batch_idx]
        kv_len = kv_lengths[batch_idx]
        for head_idx in range(n1):
            n2_idx = head_idx // group
            v_scale_value = float(v_scale[n2_idx].item())
            for qb_idx in range(qb_max):
                q_start = qb_idx * sparse_q_block_size
                if q_start >= q_len:
                    break
                q_end = min(q_start + sparse_q_block_size, q_len)
                q_indices = list(range(q_start, q_end))

                chunk_positions = _positions_from_sparse(
                    case,
                    sparse_indices,
                    sparse_seq_len,
                    batch_idx,
                    head_idx,
                    qb_idx,
                    kv_len,
                )
                if not chunk_positions:
                    continue
                positions = [p for chunk in chunk_positions for p in chunk]

                # gather q 向量（批量），shape (nq, D)
                base = int(cu_seqlens_q[batch_idx].item())
                if layout_q == "NTD":
                    q_block = query[head_idx, base + q_start : base + q_end].to(
                        torch.float32
                    )
                    q_scale_block = q_scale[head_idx, base + q_start : base + q_end].to(
                        torch.float32
                    )
                else:
                    q_block = query[base + q_start : base + q_end, head_idx].to(
                        torch.float32
                    )
                    q_scale_block = q_scale[base + q_start : base + q_end, head_idx].to(
                        torch.float32
                    )
                nq = q_block.shape[0]

                # gather k/v/k_scale（按 positions 顺序）：
                # 通过 block_table 把逻辑块映射到 kv_cache 物理块，与 kernel 读取路径一致
                pos_tensor = torch.as_tensor(positions, dtype=torch.long)
                logical_block = pos_tensor // sparse_kv_block_size
                token_in_block = pos_tensor % sparse_kv_block_size
                physical_block = block_table[batch_idx, logical_block]
                k_mat = kv_key[physical_block, n2_idx, token_in_block].to(
                    torch.float32
                )  # (npos, D)
                v_mat = kv_value[physical_block, n2_idx, token_in_block].to(
                    torch.float32
                )  # (npos, D)
                k_scale_vec = kv_k_scale[physical_block, n2_idx, token_in_block, 0].to(
                    torch.float32
                )  # (npos,)

                npos = pos_tensor.shape[0]
                # valid_mask: (nq, npos) — causal/actlen，与参考 _mask_positions 一致
                if case["mask_mode"] == 0:
                    valid_mask = torch.ones((nq, npos), dtype=torch.bool)
                elif case["mask_mode"] == 3:
                    q_idx_col = torch.as_tensor(q_indices, dtype=torch.long).view(nq, 1)
                    limit = q_idx_col + (kv_len - q_len)  # (nq,1)
                    valid_mask = pos_tensor.view(1, npos) <= limit  # (nq, npos)
                else:
                    raise ValueError(f"unsupported mask_mode: {case['mask_mode']}")

                # scores: (nq, npos)，标度在 matmul 之后施加（保持 FP8 桶一致）
                # 与 kernel 对齐乘法顺序：先将 q_scale 乘以 softmax_scale(dScale)，
                # 再乘到 QK 结果上，最后乘 k_scale，避免浮点非结合律导致的量化边界差异
                scores = torch.matmul(q_block, k_mat.transpose(0, 1))
                scores = (
                    scores
                    * (q_scale_block.view(nq, 1) * softmax_scale)
                    * k_scale_vec.view(1, npos)
                )
                scores = torch.where(
                    valid_mask, scores, torch.full_like(scores, MASK_VALUE)
                )

                # 按 2-block chunk 的 online/flash FP8 数据流，向量化到 q 维
                # p_scale 通过 ln(pScale) 编入 max（与 kernel FusedExpSub 路径对齐），
                # 使 exp 的数值路径与 kernel 一致：exp(score - (actual_max - ln(pScale))) = p_c * pScale
                # v_scale 乘入时机与 kernel 对齐：
                #   chunk 0 (DataCopy): acc = pv_raw（不含 v_scale）
                #   chunk 1 (isUpdatePre=true): acc = acc * rescale * v_scale + pv_raw * v_scale
                #   chunk >1 (isUpdatePre=false): acc = acc * rescale + pv_raw * v_scale
                #   单 chunk 最后除法 (LastDivNew): output = pv_raw * v_scale / sum
                m_run = torch.full((nq,), SOFTMAX_MAX_SENTINEL, dtype=torch.float32)
                m_run_offset = torch.full(
                    (nq,), SOFTMAX_MAX_SENTINEL, dtype=torch.float32
                )
                l_run = torch.zeros((nq,), dtype=torch.float32)
                acc = torch.zeros((nq, head_dim), dtype=torch.float32)
                offset = 0
                chunk_idx = 0
                chunk_count = len(chunk_positions)
                for chunk_pos_list in chunk_positions:
                    c0 = offset
                    c1 = offset + len(chunk_pos_list)
                    offset = c1
                    s_c = scores[:, c0:c1]
                    vm_c = valid_mask[:, c0:c1]
                    # 每行的局部 max；无有效位置时保留 kernel 使用的有限哨兵。
                    masked_scores = torch.where(
                        vm_c,
                        s_c,
                        torch.full_like(s_c, SOFTMAX_MAX_SENTINEL),
                    )
                    chunk_max = masked_scores.max(dim=-1).values  # (nq,)
                    chunk_max = torch.maximum(
                        chunk_max,
                        torch.full_like(chunk_max, SOFTMAX_MAX_SENTINEL),
                    )
                    chunk_has = chunk_max != SOFTMAX_MAX_SENTINEL
                    run_started = m_run != SOFTMAX_MAX_SENTINEL
                    # m_new：未开始的行取 chunk_max；已开始的行取历史与当前 max。
                    m_new = torch.where(
                        run_started, torch.maximum(m_run, chunk_max), chunk_max
                    )
                    # 该 chunk 对此行无有效位置 -> 此行本轮不更新
                    m_new = torch.where(chunk_has, m_new, m_run)
                    # 编入 ln(pScale)，与 kernel 的 max0 = actual_max - ln(pScale) 对齐
                    m_new_offset = m_new - ln_p_scale
                    # 与 kernel FusedExpSub 对齐：kernel 的 FusedExpSub(x, y) 计算 exp(x - y) 时，
                    # 减法在硬件内部以更高精度完成（融合，无 float32 中间舍入）。
                    # golden 用 float64 减法模拟该融合减法，避免 float32 减法舍入导致
                    # exp 结果落在 FP8 量化边界的错误一侧。
                    rescale = torch.where(
                        run_started,
                        torch.exp(
                            m_run_offset.double() - m_new_offset.double()
                        ).float(),
                        torch.zeros_like(m_run),
                    )
                    rescale = torch.where(
                        torch.isfinite(rescale), rescale, torch.zeros_like(rescale)
                    )
                    # FusedExpSub 等价：exp(score - (actual_max - ln(pScale))) = p_c * pScale
                    p_scaled = torch.exp(
                        s_c.double() - m_new_offset.double().view(nq, 1)
                    ).float()
                    p_scaled = torch.where(vm_c, p_scaled, torch.zeros_like(p_scaled))
                    # 仅对本轮有有效位置的行参与量化累加；其余行贡献 0
                    p_scaled = torch.where(
                        chunk_has.view(nq, 1), p_scaled, torch.zeros_like(p_scaled)
                    )
                    # FP8 量化：p_scaled 已含 pScale 因子，直接 cast（与 kernel 一致）
                    p_quant = p_scaled.to(fp8_dtype).to(torch.float32)
                    # BMM2：不除 pScale（与 kernel 一致，pScale 在分子分母中同时出现，数学上约掉）
                    pv_raw = torch.matmul(
                        p_quant, v_mat[c0:c1]
                    )  # (nq, D)，不含 v_scale
                    # 与 kernel FlashUpdate/FlashUpdateLast 的 isUpdatePre 逻辑对齐 v_scale 乘入时机
                    if chunk_idx == 0:
                        # 第一个 chunk：acc = pv_raw（与 kernel DataCopy 一致，不含 v_scale）
                        acc = pv_raw
                    elif chunk_idx == 1:
                        # 第二个 chunk（isUpdatePre=true）：acc * rescale * v_scale + pv_raw * v_scale
                        acc = (
                            acc * rescale.view(nq, 1) * v_scale_value
                            + pv_raw * v_scale_value
                        )
                    else:
                        # 后续 chunk（isUpdatePre=false）：acc * rescale + pv_raw * v_scale
                        acc = acc * rescale.view(nq, 1) + pv_raw * v_scale_value
                    # sum 累加含 pScale 因子的值（与 kernel 一致）
                    l_run = l_run * rescale + p_scaled.sum(dim=-1)
                    m_run = torch.where(chunk_has, m_new, m_run)
                    m_run_offset = torch.where(chunk_has, m_new_offset, m_run_offset)
                    chunk_idx += 1

                attn, lse = _finalize_fp8_attention_rows(
                    acc,
                    l_run,
                    m_run,
                    m_run_offset,
                    v_scale_value,
                    chunk_count <= 1,
                )

                for li, q_idx in enumerate(q_indices):
                    _set_output(
                        attention_out,
                        cu_seqlens_q,
                        batch_idx,
                        q_idx,
                        head_idx,
                        attn[li].to(output_dtype),
                    )
                    _set_lse(
                        softmax_lse, cu_seqlens_q, batch_idx, q_idx, head_idx, lse[li]
                    )

    return attention_out, softmax_lse


def save_test_case(input_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data["Testcase_Name"]
    input_filepath = os.path.join(output_dir, f"bsa_case_{case_name}.pt")
    torch.save(input_data, input_filepath)
    print(f"saved test case to: {input_filepath}")
    return input_filepath


def generate_and_save_testdata(params, save_pt=False, save_path=""):
    case = _normalize_params(params)
    check_valid_param.check_valid_param(case)

    rng = random.Random(case["seed"])
    generator = torch.Generator().manual_seed(case["seed"])

    batch = case["B"]
    s1 = case["S1"]
    s2 = case["S2"]
    n1 = case["N1"]
    n2 = case["N2"]
    head_dim = case["D"]
    sparse_q_block_size = case["sparse_q_block_size"]
    sparse_kv_block_size = case["sparse_kv_block_size"]

    q_lengths, q_max_len, kv_lengths, kv_max_len = _make_lengths(case)
    s1, case["S1"] = q_max_len, q_max_len
    s2, case["S2"] = kv_max_len, kv_max_len

    cu_seqlens_q = torch.tensor(case["cu_seqlens_q_value"], dtype=torch.int32)
    seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)

    cu_seqlens_q_input = cu_seqlens_q

    block_table = _make_block_table(
        batch,
        seqused_kv,
        sparse_kv_block_size,
        case["block_table_pattern"],
        rng,
        blocknum=case.get("block_num"),
        max_block_per_batch=case["max_block_per_batch"],
    )
    case["data_generation_mode"] = "fia_style_fullquant"
    case["data_range_left"] = DATA_RANGE_LEFT
    case["data_range_right"] = DATA_RANGE_RIGHT
    case["source_absmax"] = _source_absmax()
    case["raw_fp8_absmax"] = FP8_E4M3_MAX
    query, dense_key, dense_value, q_scale, dense_k_scale, v_scale, p_scale = (
        _make_fullquant_tensors(case, cu_seqlens_q, seqused_kv, generator)
    )
    kv_cache_storage, kv_cache_meta = combined_kv_cache.pack_combined_kv_cache(
        dense_key,
        dense_value,
        dense_k_scale,
        seqused_kv,
        block_table,
        sparse_kv_block_size,
        case["pa_block_padding_bytes"],
        case["layout_kv"],
        physical_block_num=case["block_num"],
    )
    combined_kv_cache.assert_combined_kv_views(kv_cache_storage, kv_cache_meta)
    atten_mask = torch.tril(torch.ones((2048, 2048), dtype=torch.uint8)).T.contiguous()
    sparse_indices, sparse_seq_len = _make_sparse_indices(
        case, q_lengths, kv_lengths, rng
    )

    if case["Testcase_Name"] is None:
        mode = "prefill"
        case["Testcase_Name"] = (
            f"quantBlockSparseAttn_{mode}_{case['layout_q']}_{case['layout_kv']}_"
            f"{batch}_{n1}_{n2}_{s1}_{s2}_{head_dim}"
        )
    case["FLOPS"], case["MFU*时间"] = _log_cube_compute_amount(
        case["Testcase_Name"],
        case,
        q_lengths,
        kv_lengths,
        sparse_indices,
        sparse_seq_len,
    )

    attention_out, softmax_lse = _reference_attention(
        case,
        query,
        kv_cache_storage,
        kv_cache_meta,
        block_table,
        q_scale,
        v_scale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        cu_seqlens_q,
        q_lengths,
        kv_lengths,
    )

    golden = {
        "attention_out": attention_out,
        "softmax_lse": softmax_lse,
    }
    input_data = {
        "Testcase_Name": case["Testcase_Name"],
        "params": case,
        "metadata_input": {
            "num_heads_q": n1,
            "num_heads_kv": n2,
            "head_dim": head_dim,
            "batch_size": batch,
            "layout_q": case["layout_q"],
            "layout_kv": case["layout_kv"],
        },
        "input": {
            "query": query,
            "kv_cache_storage": kv_cache_storage,
            "kv_cache_meta": kv_cache_meta,
            "q_descale": q_scale,
            "v_descale": v_scale,
            "p_scale": p_scale,
            "cu_seqlens_q": cu_seqlens_q_input,
            "cu_seqlens_kv": None,
            "seqused_q": None,
            "seqused_kv": seqused_kv,
            "sparse_indices": sparse_indices,
            "sparse_seq_len": sparse_seq_len,
            "block_table": block_table,
            "atten_mask": atten_mask,
            "metadata": None,
            "softmax_scale": case["softmax_scale"],
            "sparse_q_block_size": sparse_q_block_size,
            "sparse_kv_block_size": sparse_kv_block_size,
            "layout_kv": case["layout_kv"],
            "layout_q": case["layout_q"],
            "layout_sparse_indices": case["layout_sparse_indices"],
            "layout_out": case["layout_out"],
            "quant_mode": case["quant_mode"],
            "mask_mode": case["mask_mode"],
            "return_softmax_lse": case["return_softmax_lse"],
        },
        "golden": golden,
        "cpu_output": attention_out,
        "cpu_softmax_lse": softmax_lse,
    }

    if save_pt:
        save_test_case(input_data, save_path)

    return input_data
